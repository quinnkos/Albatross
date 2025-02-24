import torch
import torch.nn as nn
import torchvision.models as models


class ClusterClassificationModel(nn.Module):
    """
    CNN to classify images into geographic clusters
    """

    def __init__(self, num_clusters):
        super(ClusterClassificationModel, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, num_clusters)
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, num_images, C, H, W = x.shape
        x = x.view(batch_size * num_images, C, H, W)  # output: (batch_size*4, C, H, W)

        # Extract features
        x = self.feature_extractor(x)  # output: (batch_size*4, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)  # output: (batch_size*4, 512)

        # Average embeddings
        x = x.view(batch_size, num_images, -1)  # output: (batch_size, 4, 512)
        x = x.mean(dim=1)  # output: (batch_size, 512)

        # Classification
        x = self.fc(x)  # output: (batch_size, num_clusters)
        x = self.softmax(x)

        return x
