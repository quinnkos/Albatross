import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from geopy.distance import geodesic
from scipy.stats import gaussian_kde
from shapely.geometry import Point
from shapely.ops import nearest_points
from statistics import mean, median

# Define decay constant for geoguessr_score()
DECAY_CONSTANT = 1492.7

# Define land mask for snap_to_nearest_land()
land_mask_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
land_mask = gpd.read_file(land_mask_url)


def mean_distance(distances):
    """
    Calculate the mean distance given an array of distance values
    """
    return mean(distances)


def median_distance(distances):
    """
    Calculate the median distance given an array of distance values
    """
    return median(distances)


def geoguessr_score(distances):
    """
    Calculate the mean GeoGuessr score the AI would achieve given an array of distance values representing the
    distance between each coordinate and the AI's corresponding prediction
    """
    scores = [np.round(5000 * np.exp(-distance / DECAY_CONSTANT)) for distance in distances]
    return mean(scores)


def haversine_distance(coord1, coord2):
    """
    Calculate the haversine distance between two coordinates
    """
    return geodesic(coord1, coord2).km


def snap_to_nearest_land(lat, lon, land_mask=land_mask):
    """
    Find and return the coordinate of the nearest land given a latitude and longitude value
    """
    ocean_point = Point(lon*180, lat*90)
    nearest_land = nearest_points(ocean_point, land_mask.union_all())[1]
    return torch.tensor([nearest_land.y/90, nearest_land.x/180], dtype=torch.float32)


def denormalize_coordinates(true_coords):
    """
    Denormalize coordinate values to their original scales
    """
    denormalized_coords = []
    for lat, lon in true_coords:
        denormalized_coords.append((lat * 90, lon * 180))
    return denormalized_coords


def weighted_mean_selection(clusters, k, snap_to_land=True):
    """
    Compute the weighted mean coordinates given the top k clusters sorted by probability
    """
    # Find top k clusters and the sum of their probabilities
    top_k_clusters = sorted(clusters, key=lambda x: x[2], reverse=True)[:k]
    total_prob = sum(cluster[2] for cluster in top_k_clusters)

    # Compute weighted mean
    weighted_lat = sum(cluster[0] * (cluster[2] / total_prob) for cluster in top_k_clusters)
    weighted_lon = sum(cluster[1] * (cluster[2] / total_prob) for cluster in top_k_clusters)
    
    # If we want to adjust all weighted coordinates to be on land
    if snap_to_land:
        return snap_to_nearest_land(weighted_lat, weighted_lon)

    return weighted_lat, weighted_lon


def weighted_median_selection(clusters, k):
    """
    Compute the weighted median coordinates given the top k clusters sorted by probability
    """
    # Find top k clusters and the sum of their probabilities (computer cumulative weights for later)
    top_k_clusters = sorted(clusters, key=lambda x: x[2], reverse=True)[:k]
    cum_weights = np.cumsum([cluster[2] for cluster in top_k_clusters])
    total_prob = cum_weights[-1]

    # Sort clusters by latitude and longitude separately
    clusters_by_lat = sorted(top_k_clusters, key=lambda x: x[0])
    clusters_by_lon = sorted(top_k_clusters, key=lambda x: x[1])

    # Compute weighted median
    weighted_lat = next(cluster[0] for cluster, weight in zip(clusters_by_lat, cum_weights) if weight >= total_prob/2)
    weighted_lon = next(cluster[1] for cluster, weight in zip(clusters_by_lon, cum_weights) if weight >= total_prob/2)

    return weighted_lat, weighted_lon


def get_kde_centroids(group):
    """
    Compute cluster centroids based on the geographic density of the datapoints 
    """
    latitudes = group['latitude'].values
    longitudes = group['longitude'].values

    # If only one point in cluster
    if len(latitudes) < 2:
        return pd.Series({'latitude': latitudes[0], 'longitude': longitudes[0]})

    # Perform KDE
    kde = gaussian_kde(np.vstack([latitudes, longitudes]))

    # Evaluate KDE on the original points
    densities = kde(np.vstack([latitudes, longitudes]))

    # Get the coordinates corresponding to the highest density
    max_density_idx = np.argmax(densities)
    kde_lat, kde_lon = latitudes[max_density_idx], longitudes[max_density_idx]

    return pd.Series({'latitude': kde_lat, 'longitude': kde_lon})


def get_centroids(df, method):
    """
    Compute geographic cluster centroids
    :param df: dataframe containing 'cluster' column
    :param method: if 'mean', then return the geographic mean of each cluster; 
                    if 'kde', then return the density-based cluster centroid
    """
    if method == "mean":
        return df.groupby('cluster')[['latitude', 'longitude']].mean().astype(float)
    elif method == "kde":
        return df.groupby('cluster').apply(get_kde_centroids)
    else:
        raise ValueError("Invalid method. Use 'mean' or 'kde'.")
