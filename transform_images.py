import cv2
import numpy as np
import pandas as pd

# Load panoramas dataset
df = pd.read_csv("df_panoramas.csv")
df = df[~df['image_path'].str.endswith('/nan')]


def crop_to_360(panorama):
    """
    Crop panorama to 360 degrees such that width = 2 * height
    :param panorama: a panorama from df_panoramas.csv
    :return: the cropped panorama
    """
    panorama = np.array(panorama)
    height, _ = panorama.shape[:2]
    cropped_panorama = panorama[:, :(2 * height)]

    return cropped_panorama


def panorama_to_image(panorama, pitch, yaw, fov, output_width, output_height):
    """
    :param panorama: a panorama from df_panoramas.csv
    :param pitch: the up-down rotation to apply to transform the panorama
    :param yaw: the left-right rotation to apply to transform the panorama
    :param fov: the field of view to capture
    :param output_width: the width of the transformed image
    :param output_height: the height of the transformed image
    :return: the transformed image
    """
    panorama = np.array(panorama)
    panorama_height, panorama_width = panorama.shape[:2]

    fov = np.radians(fov)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Create grid of pixel coordinates
    x, y = np.meshgrid(np.arange(output_width), np.arange(output_height))

    # Normalize pixel coordinates and stack into a 3xN array
    focal_length = output_width / (2 * np.tan(fov / 2))
    nx = (x - output_width / 2) / focal_length
    ny = (y - output_height / 2) / focal_length
    nz = np.full_like(nx, -1)
    vectors = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3).T

    # Compute rotation matrix (combined X and Y rotations)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_y, 0, sin_y],
        [sin_p * sin_y, cos_p, -sin_p * cos_y],
        [-cos_p * sin_y, sin_p, cos_p * cos_y]
    ])

    # Apply rotation and use components to compute longitude and latitude
    rotated_vectors = rotation_matrix @ vectors
    rx, ry, rz = rotated_vectors
    longitude = np.arctan2(rx, -rz)
    latitude = np.arctan2(ry, np.sqrt(rx ** 2 + rz ** 2))

    # Map to panorama coordinates and reshape to intended dimensions
    pan_x = ((longitude + np.pi) / (2 * np.pi) * panorama_width).astype(int) % panorama_width
    pan_y = ((latitude + np.pi / 2) / np.pi * panorama_height).astype(int) % panorama_height
    pan_x = pan_x.reshape(output_height, output_width)
    pan_y = pan_y.reshape(output_height, output_width)

    # Assign pixel data and return transformed image
    new_image = panorama[pan_y, pan_x]
    return new_image


# Set parameters
FOV = 90
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
PITCH = [0, 0, 0, 0]
YAW = [0, 90, -90, 180]
IMAGE_LABELS = ["center", "left", "right", "back"]

image_paths = [[], [], [], []]

# Apply transformation to each panorama
for idx, row in df.iterrows():
    if isinstance(row['image_path'], (int, float)):
        break

    if idx % 1000 == 0:
        print(f"Processing image {idx} of {len(df)}")

    panorama = cv2.imread(row['image_path'])
    panorama = crop_to_360(panorama)

    # Convert panorama into four images and store the new image paths
    for i in range(len(IMAGE_LABELS)):
        new_image = panorama_to_image(panorama, PITCH[i], YAW[i], FOV, IMAGE_WIDTH, IMAGE_HEIGHT)
        save_path = f"{row['image_path']}_{IMAGE_LABELS[i]}.jpeg"
        cv2.imwrite(save_path, new_image)
        image_paths[i].append(save_path)

# Add image path lists to the dataframe as columns
for i in range(len(IMAGE_LABELS)):
    print(image_paths[i])
    df[f'image_path_{IMAGE_LABELS[i]}'] = image_paths[i]

# Save updated dataframe as a csv for later use
df = df.drop(columns=["Unnamed: 0", "image_path"])
df.to_csv("/content/drive/MyDrive/df_streetview_images_processed.csv")
