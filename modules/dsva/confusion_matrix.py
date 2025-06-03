import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def compute_confusion_matrix(model, dataloader, device, num_classes):
    """
    Run inference on the given dataloader and compute the confusion matrix.

    Args:
        model (torch.nn.Module): trained classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader for ModelNet40 test set.
        device (torch.device): torch device (e.g., torch.device('cuda')).
        num_classes (int): number of classes (40 for ModelNet40).
    Returns:
        cm (np.ndarray): confusion matrix shape (num_classes, num_classes),
                         where cm[i, j] = count of true label i predicted as j.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in dataloader:
            # points: [B, N, 3] or however your dataset returns them
            # labels: [B]
            points = points.to(device)           # e.g., [B, 1024, 3]
            labels = labels.to(device)           # [B]

            # Forward pass
            logits = model(points)               # assume output shape [B, num_classes]
            preds = logits.argmax(dim=1)         # [B]

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # shape [num_samples]
    all_labels = np.concatenate(all_labels, axis=0) # shape [num_samples]

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm

def plot_confusion_matrix(cm, class_names, normalize=True, figsize=(10, 8), cmap=plt.cm.Blues):
    """
    Plot and display the confusion matrix.

    Args:
        cm (np.ndarray): confusion matrix (num_classes x num_classes).
        class_names (List[str]): list of class names of length num_classes.
        normalize (bool): if True, normalize each row to sum to 1.
        figsize (tuple): size of the matplotlib figure.
        cmap: matplotlib colormap for the heatmap.
    """
    if normalize:
        # Avoid division by zero: only normalize rows with sum > 0
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, cm_sum, where=cm_sum != 0)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar(fraction=0.046, pad=0.04)

    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        val = cm[i, j]
        plt.text(
            j, i,
            format(val, fmt),
            horizontalalignment="center",
            color="white" if val > thresh else "black",
            fontsize=5
        )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

# Example usage:
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from your_modelnet40_dataset import ModelNet40Dataset  # replace with your dataset class
    from your_model_definition import YourModel            # replace with your model

    # 1. Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 40

    model = YourModel(num_classes=num_classes)
    model.load_state_dict(torch.load("path/to/your_trained_model.pth"))
    model.to(device)

    # 2. Prepare test DataLoader
    test_dataset = ModelNet40Dataset(split='test', num_points=1024)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. (Optional) Define class names in the same order as labels (0–39)
    class_names = [
        'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',
        'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',
        'laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink',
        'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox'
    ]

    # 4. Compute confusion matrix
    cm = compute_confusion_matrix(model, test_loader, device, num_classes)

    # 5. Plot and save
    plot_confusion_matrix(cm, class_names, normalize=True, figsize=(12, 10))
    plt.savefig("modelnet40_confusion_matrix.png", dpi=300)
    plt.show()
