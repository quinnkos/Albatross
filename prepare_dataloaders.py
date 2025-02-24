import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split

# Load transformed & clustered images dataset
df = pd.read_csv('/content/drive/MyDrive/df_streetview_image_sets_gmmclustered_k200.csv')

# Normalize latitude and longitude to values in [-1.0, 1.0]
df['latitude'] = df['latitude'] / 90
df['longitude'] = df['longitude'] / 180

# Ensure cluster indices are contiguous
cluster_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(df['cluster'].unique())}
df['cluster'] = df['cluster'].map(cluster_idx_map)

# Train/val/test split (70%/15%/15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    """
    :return: 4-image set, cluster index, and coordinates of each item
    """
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Transform images
        center_image = Image.open(row['image_path_center'])
        left_image = Image.open(row['image_path_left'])
        right_image = Image.open(row['image_path_right'])
        back_image = Image.open(row['image_path_back'])
        center_image = self.transform(center_image)
        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        back_image = self.transform(back_image)

        # Define outputs
        image_set = torch.stack([center_image, left_image, right_image, back_image], dim=0)
        cluster = self.df.iloc[idx]['cluster']
        coords = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)

        return image_set, cluster, coords


# Create dataloaders for train, val, and test
train_dataset = ImageDataset(train_df, transform=transform)
val_dataset = ImageDataset(val_df, transform=transform)
test_dataset = ImageDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
