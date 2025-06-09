import numpy as np
import sklearn.metrics as metrics
import wandb


class StatsMixin:
    """
    Mixin providing methods to create stats
    """

    def save_misclassified(self, coords, label, mis_examples, preds):
        # save misclassified examples
        if len(mis_examples) < 5:
            diff = (preds != label).nonzero(as_tuple=False).squeeze(1)
            for b in diff.tolist():
                if len(mis_examples) >= 5:
                    break
                pc = coords[b].cpu().numpy().T
                mis_examples.append((pc, label[b].item(), preds[b].item()))


    def check_stats(
            self,
            count,
            epoch,
            test_loss,
            test_pred,
            test_true,
            avg_train_loss,
            best_test_acc
    ):
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if float(test_acc) < float(best_test_acc):
            test_acc_str = f"TestAcc=🔥{test_acc:.4f}🔥"
        else:
            test_acc_str = f"TestAcc=🔥{test_acc:.4f}🔥✅"

        outstr = (
            f"Epoch {epoch + 1:3d}/{self.args.epochs:3d} "
            f"TrainAvgLoss=🔥{avg_train_loss:.4f}🔥 "
            f"TestLoss={(test_loss / count):.4f} "
            f"TestAvgPerClassAcc={avg_per_class_acc:.4f} "
            f"{test_acc_str} "
        )

        print(outstr)

        if self.args.wandb:
            wandb.log({
                "test/TestAcc": test_acc,
                "test/TestLoss": (test_loss / count),
                "test/TestAvgPerClassAcc": avg_per_class_acc,
                "epoch": epoch
            })

        return test_acc

    def log_gradient_and_param_statistics(
            self,
            epoch=None,
            step=None,
            detect_anomalies=True
    ):
        import wandb
        grad_stats = {}
        param_stats = {}
        global_step = step if step is not None else epoch

        total_grad_norm_sq = 0.0
        total_param_norm_sq = 0.0
        vanishing_layers = []
        exploding_layers = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                param_data = param.detach().float()

                # Per-layer gradient stats
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                grad_mean = grad.mean().item()
                grad_median = grad.median().item()
                grad_std = grad.std().item()
                grad_norm = grad.norm(2).item()
                total_grad_norm_sq += grad_norm ** 2

                grad_stats[f"grad/{name}/min"] = grad_min
                grad_stats[f"grad/{name}/max"] = grad_max
                grad_stats[f"grad/{name}/mean"] = grad_mean
                grad_stats[f"grad/{name}/median"] = grad_median
                grad_stats[f"grad/{name}/std"] = grad_std
                grad_stats[f"grad/{name}/l2_norm"] = grad_norm

                # Optional anomaly detection
                if detect_anomalies:
                    if grad_norm < 1e-6:
                        vanishing_layers.append(name)
                    if grad_norm > 1e3:
                        exploding_layers.append(name)

                # Per-layer parameter stats (optional)
                param_stats[f"param/{name}/l2_norm"] = param_data.norm(2).item()
                param_stats[f"param/{name}/mean"] = param_data.mean().item()
                total_param_norm_sq += param_data.norm(2).item() ** 2

        # Global stats
        grad_stats["grad/global_l2_norm"] = total_grad_norm_sq ** 0.5
        param_stats["param/global_l2_norm"] = total_param_norm_sq ** 0.5

        # Logging to wandb
        wandb.log({**grad_stats, **param_stats}, step=global_step)

        # Print anomaly diagnostics
        if detect_anomalies:
            if vanishing_layers:
                print(f"🟡 Vanishing gradients detected in: {vanishing_layers}")
            if exploding_layers:
                print(f"🔴 Exploding gradients detected in: {exploding_layers}")
