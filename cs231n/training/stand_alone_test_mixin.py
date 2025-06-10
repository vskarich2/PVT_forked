import numpy as np
import torch
from sklearn import metrics
from tqdm.auto import tqdm


class StandAloneTestMixin:

    def stand_alone_test(self):
        self.test_loader = self.get_test_loader()

        print(f"\nPure Testing Run Starting with....{self.args.dataset}")
        # Fill in these values to make sure stats logging works ok
        train_avg_loss = 0.0
        best_test_acc = 0.0

        epoch = 0
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        mis_examples = []
        self.model.eval()

        test_bar = tqdm(
            self.test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/1)",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label, *maybe_class_name in test_bar:
                class_name = maybe_class_name[0] if maybe_class_name else "NONE"

                (feats, coords), label = self.preprocess_test_data(data, label)
                feats = feats.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                logits = self.model(feats)
                loss = self.criterion(logits, label)
                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                self.save_misclassified(coords, label, mis_examples, preds)

            count += 1.0
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.cpu().numpy())

            if self.args.compute_saliency and self.args.wandb:
                self.log_saliency(epoch)

            running_avg_loss = test_loss / count
            test_bar.set_postfix(test_loss=running_avg_loss)

        test_bar.close()

        # ───────────────────────────────────────────────────────────────
        # Confusion matrix and W&B logging
        # ───────────────────────────────────────────────────────────────
        if self.args.conf_matrix and self.args.wandb:
            self.log_confusion_matrix(test_true, test_pred, epoch)
            self.log_misclassified(mis_examples, epoch)
            self.log_per_class_accuracies(test_true, test_pred, epoch)

        # Final scalar metrics
        test_acc = self.check_stats(
            count,
            epoch,
            test_loss,
            test_pred,
            test_true,
            train_avg_loss,
            best_test_acc
        )

        return test_acc