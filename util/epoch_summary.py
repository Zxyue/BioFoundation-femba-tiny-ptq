import pytorch_lightning as pl
from tqdm.auto import tqdm

class EpochSummary(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        def get_value(name):
            value = metrics.get(name)
            if value is None:
                return "NA"
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            return f"{value:.6f}"

        msg = (
            f"[epoch {trainer.current_epoch:02d}] "
            f"train_loss={get_value('train_loss_epoch')} "
            f"val_loss={get_value('val_loss')} "
            f"val_AUROC={get_value('val_BinaryAUROC')} "
            f"val_F1={get_value('val_BinaryF1Score')}"
        )

        # tqdm.write 会在进度条上方打印，不破坏进度条
        tqdm.write(msg)