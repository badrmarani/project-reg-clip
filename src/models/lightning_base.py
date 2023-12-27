import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy

from ..metrics import WorstGroupAccuracy


def get_device(model: nn.Module):
    return next(model.parameters()).device


class LightningBase(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer_cfg):
        super().__init__()

        self.model = model
        self.optimizer_cfg = optimizer_cfg

        device = get_device(model)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.avg_accuracy = BinaryAccuracy().to(device)
        self.wg_accuracy = WorstGroupAccuracy().to(device)

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_cfg)
        return optimizer

    def training_step(self, batch, batch_idx):
        filename = batch.pop("filename")
        logits = self.model(batch["image"])
        y_true = batch["label"].float()
        if y_true.ndim < 2:
            y_true.unsqueeze_(1)

        if logits.size(1) == 2:
            y_pred = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
        else:
            y_pred = (torch.sigmoid(logits) >= 0.5).float()

        loss = self.loss_fn(logits, y_true)

        self.log(
            "train/bce_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "train/acc_avg",
            self.avg_accuracy(y_pred, y_true),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        g = batch["group"]
        wg_acc, results = self.wg_accuracy(y_pred, y_true, g)
        self.log(
            "train/wg_acc",
            wg_acc.item(),
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in results.items():
            self.log(
                f"train/acc_group_{k}",
                v.item(),
                on_step=True,
                logger=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def on_train_epoch_end(self):
        self.avg_accuracy.reset()
        self.wg_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        filename = batch.pop("filename")
        logits = self.model(batch["image"])
        y_true = batch["label"].float()
        if y_true.ndim < 2:
            y_true.unsqueeze_(1)

        if logits.size(1) == 2:
            y_pred = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
        else:
            y_pred = (torch.sigmoid(logits) >= 0.5).float()

        loss = self.loss_fn(logits, y_true)

        self.log(
            "val/bce_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.avg_accuracy.update(y_pred, y_true)

        g = batch["group"]
        self.wg_accuracy.update(y_pred, y_true, g)

    def on_validation_epoch_end(self):
        self.log(
            "val/acc_avg",
            self.avg_accuracy.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        wg_acc, results = self.wg_accuracy.compute()
        self.log(
            "val/wg_acc",
            wg_acc.item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in results.items():
            self.log(
                f"val/acc_group_{k}",
                v.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.avg_accuracy.reset()
        self.wg_accuracy.reset()
