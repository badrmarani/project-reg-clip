import pytorch_lightning as pl
import torch
from timm import create_model
from torch import nn
from torchmetrics.classification import BinaryAccuracy

from ..metrics import WorstGroupAccuracy


class ResNet(pl.LightningModule):
    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(hparams)
        self.model = create_model(
            model_name=hparams.model_name,
            num_classes=hparams.num_classes,
            pretrained=hparams.pretrained,
        )

        self.optimizer_config = {
            "optimizer_name": hparams.optimizer_name,
            "lr": hparams.lr,
            "weight_decay": hparams.weight_decay,
            "momentum": hparams.momentum,
        }

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()
        self.train_wga = WorstGroupAccuracy()
        self.valid_wga = WorstGroupAccuracy()

    def forward(self, batch, validation: bool = False):
        batch.pop("filename")
        x = batch["image"]
        g = batch["group"]
        y_true = batch["label"].float()
        if y_true.ndim < 2:
            y_true.unsqueeze_(1)

        logits = self.model(x)
        if logits.size(1) < 2:
            y_pred = (torch.sigmoid(logits) >= 0.5).float()
        else:
            y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).unsqueeze(1)

        loss = self.loss_fn(logits, y_true)
        self.log(
            f"{self.logging_prefix}/loss",
            loss,
            prog_bar=self.log_progress_bar,
            logger=True,
            sync_dist=True,
        )

        if validation:
            self.valid_accuracy.update(y_pred, y_true)
            self.valid_wga.update(y_pred, y_true, g)
        else:
            self.train_accuracy.update(y_pred, y_true)
            self.train_wga.update(y_pred, y_true, g)

            avg_acc = self.train_accuracy.compute()
            self.log(
                f"{self.logging_prefix}/step-avg_acc",
                avg_acc,
                on_step=self.log_on_step,
                on_epoch=self.log_on_epoch,
                logger=True,
            )

            wga, acc_groups = self.train_wga.compute()
            self.log(
                f"{self.logging_prefix}/step-wga",
                wga,
                on_step=self.log_on_step,
                logger=True,
            )
            for k, v in enumerate(acc_groups):
                self.log(
                    f"{self.logging_prefix}/step-acc_grp{k}",
                    v,
                    on_step=self.log_on_step,
                    logger=True,
                )
        return loss

    def training_step(self, batch, batch_idx):
        self.logging_prefix = "train"
        self.log_on_step = True
        self.log_on_epoch = False
        loss = self(batch)
        self.log_on_step = None
        self.log_on_epoch = None
        self.logging_prefix = None
        return loss

    def on_train_epoch_end(self):
        self.log(
            "val/epoch-avg_acc",
            self.train_accuracy.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        wga, acc_groups = self.train_wga.compute()
        self.log(
            "train/epoch-wga",
            wga,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for k, v in enumerate(acc_groups):
            self.log(
                f"train/epoch-acc_grp{k}",
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.train_accuracy.reset()
        self.train_wga.reset()

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(batch, validation=True)
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val/epoch-avg_acc",
            self.valid_accuracy.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        wga, acc_groups = self.valid_wga.compute()
        self.log(
            "val/epoch-wga",
            wga,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for k, v in enumerate(acc_groups):
            self.log(
                f"val/epoch-acc_grp{k}",
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.valid_accuracy.reset()
        self.valid_wga.reset()

    @property
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        num_devices = max(1, self.trainer.num_devices)
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = (
            dataset_size
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches * num_devices)
        )
        return num_steps

    def configure_optimizers(self):
        optimizer_name = self.optimizer_config.pop("optimizer_name", "AdamW")
        optimizer = getattr(torch.optim, optimizer_name)
        optimizer = optimizer(self.model.parameters(), **self.optimizer_config)

        # t = [1] * self.num_training_steps
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lambda x: t.__getitem__(x)
        # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return [optimizer]
