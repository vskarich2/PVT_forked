import numpy as np
import matplotlib.pyplot as plt
import itertools
import itertools

import matplotlib.pyplot as plt
import numpy as np


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
