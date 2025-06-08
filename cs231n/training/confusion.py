from __future__ import print_function

import warnings

import torch
from matplotlib import pyplot as plt
# ignore everything
from tqdm.auto import tqdm

from cs231n.dsva.confusion_matrix import plot_confusion_matrix

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.metrics as metrics


class ConfusionMatrixMixin:
    """
    Mixin providing a method to compute the confusion matrix for a classification model.
    """
    def make_confusion_matrix_for_modelnet(self):

        print("Generating Confusion Matrix for ModelNet40....")

        test_loader = self.get_test_loader()
        self.model.eval()

        all_preds = []
        all_labels = []
        test_true = []
        test_pred = []

        # Standard ModelNet40 class names in label order
        class_names = [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
            "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
            "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
            "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
            "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
            "wardrobe", "xbox"
        ]

        # Will collect up to 5 wrongly classified examples: (true_name, predicted_name)
        wrong_examples = []

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label in logging_wrapper:
                    # Preprocess exactly as training
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    feats = feats.to(self.device)
                    label = label.to(self.device)

                    # Forward pass
                    logits = self.model(feats)
                    preds = logits.argmax(dim=1)

                    # Accumulate for confusion matrix
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(label.cpu().numpy())

                    # Check up to 5 misclassifications
                    if len(wrong_examples) < 5:
                        preds_cpu = preds.detach().cpu()
                        label_cpu = label.cpu()
                        for i in range(label_cpu.size(0)):
                            if len(wrong_examples) >= 5:
                                break
                            true_idx = label_cpu[i].item()
                            pred_idx = preds_cpu[i].item()
                            if pred_idx != true_idx:
                                true_name = class_names[true_idx]
                                pred_name = class_names[pred_idx]
                                wrong_examples.append((true_name, pred_name))

                # Concatenate after all batches
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # Compute and print metrics
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = f"Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}"
                print(outstr)

                # Plot confusion matrix
                print("Calculating Confusion Matrix...")
                cm = confusion_matrix(all_labels, all_preds, labels=list(range(40)))
                plot_confusion_matrix(cm, class_names)
                plt.savefig("modelnet40_confusion_matrix.png", dpi=300)
                plt.show()

        # Print up to 5 misclassified examples
        print("\nUp to 5 misclassified examples (true → predicted):")
        for idx, (true_name, pred_name) in enumerate(wrong_examples):
            print(f"  {idx + 1}. {true_name} → {pred_name}")
        if not wrong_examples:
            print("  (No misclassifications found!)")

    def make_confusion_matrix_for_scanobject(self):

        print("Generating Confusion Matrix....")

        test_loader = self.get_test_loader()
        self.model.eval()

        all_preds = []
        all_labels = []
        test_true = []
        test_pred = []

        class_names = [
            "bag", "bin", "box", "cabinet", "chair",
            "desk", "display", "door", "shelf", "table",
            "bed", "pillow", "sink", "sofa", "toilet"
        ]

        # Will collect up to 5 wrongly classified examples:
        wrong_examples = []  # list of tuples: (true_name, predicted_name)

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label, classname in logging_wrapper:
                    # Preprocess batch exactly as training
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    # Make sure feats & label are on the correct device
                    feats = feats.to(self.device)
                    label = label.to(self.device)

                    # Forward pass
                    logits = self.model(feats)
                    preds = logits.argmax(dim=1)

                    # Accumulate for confusion matrix
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(label.cpu().numpy())

                    # Check for misclassifications in this batch (up to 5 total)
                    if len(wrong_examples) < 5:
                        # Move preds & label to CPU for easy comparison
                        preds_cpu = preds.detach().cpu()
                        label_cpu = label.cpu()
                        # `classname` is a list of true class names for this batch
                        for i in range(label_cpu.size(0)):
                            if len(wrong_examples) >= 5:
                                break
                            true_idx = label_cpu[i].item()
                            pred_idx = preds_cpu[i].item()
                            if pred_idx != true_idx:
                                true_name = classname[i]  # true class name from dataset
                                pred_name = class_names[pred_idx]
                                wrong_examples.append((true_name, pred_name))

                # Concatenate everything once all batches are done
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # Compute and print overall metrics
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = f"Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}"
                print(outstr)

                # Plot confusion matrix
                cm = confusion_matrix(all_labels, all_preds, labels=list(range(15)))
                plot_confusion_matrix(cm, class_names)
                plt.savefig("scanobjectnn_confusion_matrix.png", dpi=300)
                plt.show()

        # After looping through all batches, print up to 5 misclassified examples
        print("\nUp to 5 misclassified examples (true → predicted):")
        for idx, (true_name, pred_name) in enumerate(wrong_examples):
            print(f"  {idx + 1}. {true_name} → {pred_name}")
        if not wrong_examples:
            print("  (No misclassifications found!)")


    def compute_confusion_matrix(self, model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 num_classes: int) -> np.ndarray:
        """
        Run inference on the given dataloader and compute the confusion matrix.

        Args:
            model (torch.nn.Module): trained classification model.
            dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
            device (torch.device): torch device (e.g., torch.device('cuda')).
            num_classes (int): number of classes.
        Returns:
            cm (np.ndarray): confusion matrix of shape (num_classes, num_classes),
                             where cm[i, j] = count of true label i predicted as j.
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in dataloader:
                # Move inputs and labels to the specified device
                points = points.to(device)   # e.g., [B, 1024, 3]
                labels = labels.to(device)   # [B]

                # Forward pass
                logits = model(points)        # assume output shape [B, num_classes]
                preds = logits.argmax(dim=1)  # [B]

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)   # shape [num_samples]
        all_labels = np.concatenate(all_labels, axis=0) # shape [num_samples]

        cm = confusion_matrix(y_true=all_labels, y_pred=all_preds, labels=list(range(num_classes)))

        return cm
