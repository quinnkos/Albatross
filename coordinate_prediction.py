import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch

from prepare_dataloaders import df
from utils import *


class CoordinatePredictor:
    """
    Class to define prediction methods and access shared functions
    """
    def __init__(self, method, k=5):
        self.method = method
        self.k = k
        self.pred_coords = []
        self.centroids = get_centroids(df, method="mean")

    def predict_coordinate(self, logits):
        """
        Predict coordinates according to logits and selected method
        """
        if self.method == "cluster_centroid":
            pred_cluster = torch.argmax(logits, dim=1)
            pred_coord_batch = self.centroids.loc[pred_cluster.cpu().numpy()]
            pred_coord_batch = pred_coord_batch.itertuples(index=False, name=None)

            self.pred_coords.extend(list(pred_coord_batch))

        elif self.method == "top_k_weighted_median":
            pred_coord_batch = []
            for logit in logits:
                cluster_probs = [(self.centroids.loc[i, 'latitude'], self.centroids.loc[i, 'longitude'], prob) for
                                 i, prob in enumerate(logit.cpu().numpy())]
                pred_coord_batch.append(weighted_median_selection(cluster_probs, k=self.k))
                pred_coord_batch = [tuple(coord) for coord in pred_coord_batch]

            self.pred_coords.extend(pred_coord_batch)

        elif self.method == "top_k_weighted_mean":
            pred_coord_batch = []
            for logit in logits:
                cluster_probs = [(self.centroids.loc[i, 'latitude'], self.centroids.loc[i, 'longitude'], prob) for
                                 i, prob in enumerate(logit.cpu().numpy())]
                pred_coord_batch.append(weighted_mean_selection(cluster_probs, k=self.k))
                pred_coord_batch = [tuple(coord) for coord in pred_coord_batch]

            self.pred_coords.extend(pred_coord_batch)

    def plot_predictions(self, true_coords, num_samples=32):
        """
        Plot predicted coordinates vs true coordinates on world map
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.stock_img()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        true_lons = [coord[1] for coord in true_coords[:num_samples]]
        true_lats = [coord[0] for coord in true_coords[:num_samples]]
        pred_lons = [coord[1] for coord in self.pred_coords[:num_samples]]
        pred_lats = [coord[0] for coord in self.pred_coords[:num_samples]]

        ax.scatter(true_lons, true_lats, color="blue", label='True')
        ax.scatter(pred_lons, pred_lats, color="red", label="Predicted")

        for i in range(len(true_lons)):
            ax.plot([true_lons[i], pred_lons[i]], [true_lats[i], pred_lats[i]], color='red', linestyle='--',
                    linewidth=1)

        plt.title("Predicted vs True Locations")
        plt.legend()
        plt.show()

    def print_predictions(self, true_coords):
        """
        Print summary of coordinate prediction accuracy
        """
        distances = [haversine_distance(true, pred) for true, pred in zip(true_coords, self.pred_coords)]
        print(f"Method: {self.method}")
        print(f"Mean distance difference: {mean_distance(distances):.2f} km")
        print(f"Median distance difference: {median_distance(distances):.2f} km")
        print(f"Mean GeoGuessr score: {geoguessr_score(distances)}")
        print("\n")
