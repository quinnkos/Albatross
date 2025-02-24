import torch

from coordinate_prediction import CoordinatePredictor
from prepare_dataloaders import test_loader
from train import device, model
from utils import *


def evaluate(dataloader, predictors):
    """
    Evaluate the classification model
    :param dataloader: The Dataloader object to use for evaluation
    :param predictors: A list of methods to test to convert classification outputs into geographic coordinates
    """

    if len(predictors) == 0:
        print("Invalid call: 'predictors' list is empty!")
        return

    model.eval()
    
    # Collect true coordinates corresponding to image sets in dataloader
    true_coords = []
    with torch.inference_mode():
        for batch_idx, (image_set_batch, _, true_coord_batch) in enumerate(dataloader):
            # if batch_idx >= num_batches:
            #     break

            image_set_batch = image_set_batch.to(device)
            logits = model(image_set_batch)
            
            # Collect predicted coordinates corresponding to image sets according to each prediction method
            for predictor in predictors:
                predictor.predict_coordinate(logits)

            true_coords.extend(true_coord_batch.cpu().numpy())
    
    # Denormalize predicted and true coordinate values
    for predictor in predictors:
        predictor.pred_coords = denormalize_coordinates(predictor.pred_coords)
    true_coords = denormalize_coordinates(true_coords)

    # Call plotting and printing member functions
    for predictor in predictors:
        predictor.plot_predictions(true_coords)
        predictor.print_predictions(true_coords)


# Define predictor objects
cluster_centroid_predictor = CoordinatePredictor(method="cluster_centroid")
top_k_weighted_mean_predictor = CoordinatePredictor(method="top_k_weighted_mean")
top_k_weighted_median_predictor = CoordinatePredictor(method="top_k_weighted_median")
predictors = [cluster_centroid_predictor,
              top_k_weighted_mean_predictor,
              top_k_weighted_median_predictor]

evaluate(dataloader=test_loader, predictors=predictors)
