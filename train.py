import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter

from cluster_classification import ClusterClassificationModel
from prepare_dataloaders import df, train_loader, val_loader

# Define hyperparameters
num_epochs = 10
learning_rate = 5e-5
weight_decay = 1e-5


def get_cluster_weights():
    """
    Get tensor of cluster weights for weighted cross-entropy loss
    """
    cluster_counts = Counter(df['cluster'])
    cluster_weights = [len(df) / (len(df) * count) for count in cluster_counts.values()]
    return torch.tensor(cluster_weights, dtype=torch.float32).to(device)


# Define model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clusters = df['cluster'].nunique()
model = ClusterClassificationModel(num_clusters).to(device)
weights = get_cluster_weights()
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(num_epochs):
    
    # Manually update optimization settings while training
    # optimizer = optim.Adam(model.parameters(), lr=___, weight_decay=___)

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Training
    model.train()
    for image_set, true_cluster, _ in train_loader:
        image_set, true_cluster = image_set.to(device), true_cluster.to(device).type(torch.long)

        optimizer.zero_grad()
        logits = model(image_set)
        loss = loss_fn(logits, true_cluster)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred_cluster = torch.argmax(logits, dim=1)
        train_correct += (pred_cluster == true_cluster).sum().item()
        train_total += true_cluster.size(0)

    # Validation
    model.eval()
    with torch.inference_mode():
        for image_set, true_cluster, _ in val_loader:
            image_set, true_cluster, true_coords = image_set.to(device), true_cluster.to(device).type(
                torch.long), true_coords.to(device)
            logits = model(image_set)
            loss = loss_fn(logits, true_cluster)
            val_loss += loss.item()

            pred_cluster = torch.argmax(logits, dim=1)
            val_correct += (pred_cluster == true_cluster).sum().item()
            val_total += true_cluster.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    generalization_gap = (train_acc - val_acc) / train_acc

    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Acc: {val_acc:.4f}", 
          f"Generalization Gap: {generalization_gap}")

    # Update learning rate and/or weight decay, if necessary
    '''
    if generalization_gap > ___:
        learning_rate /= ___
        weight_decay *= ___
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    '''
