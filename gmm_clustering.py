import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Set parameters
num_clusters = 500

# Load transformed images dataset
df = pd.read_csv("df_streetview_images_processed.csv")
coordinates = df[['latitude', 'longitude']].values

# Assign clusters to images
gmm = GaussianMixture(n_components=num_clusters, random_state=42, covariance_type='full')
gmm.fit(coordinates)
df['cluster'] = gmm.predict(coordinates)


# ---PLOTTING---

# Define map boundaries
x_min, x_max = df['longitude'].min() - .01, df['longitude'].max() + .01
y_min, y_max = df['latitude'].min() - .01, df['latitude'].max() + .01

# Create Cartopy map
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Create mesh grid for decision boundary
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_coordinates = np.c_[yy.ravel(), xx.ravel()]
Z = gmm.predict(grid_coordinates).reshape(xx.shape)

# Plot cluster regions
plt.pcolormesh(xx, yy, Z, alpha=0.5, cmap='tab20', shading='auto', transform=ccrs.PlateCarree())

'''
# Scatter plot of actual data points
for cluster in range(num_clusters):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'], s=10, transform=ccrs.PlateCarree())
'''

# Show plot
plt.title('GMM Cluster Borders on Cartopy')
plt.show()
