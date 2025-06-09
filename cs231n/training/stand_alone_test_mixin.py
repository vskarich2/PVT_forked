import numpy as np
import torch
from sklearn import metrics
from tqdm.auto import tqdm


class StandAloneTestMixin:
    def stand_alone_test(self):

        print(f"\nPure Testing Run Starting with....{self.args.dataset}")

        test_loader = self.get_test_loader()

        self.model.eval()
        test_true = []
        test_pred = []

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label in logging_wrapper:
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    feats = feats.to(self.device, non_blocking=True)
                    coords = coords.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)

                    logits = self.model(feats)
                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
                print(outstr)
