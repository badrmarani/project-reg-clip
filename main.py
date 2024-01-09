import json
import os
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from src.datasets import CelebADataModule
from src.models.resnet import ResNet


def main(hparams):
    use_cuda = torch.cuda.is_available()

    if hparams.dataset == "celeba":
        DataModule = CelebADataModule
    else:
        raise ValueError(f"The dataset {hparams.dataset} is not available.")

    seed = random.randint(1e5, 9e5)  # type: ignore
    experiment_name = hparams.dataset + "_" + str(seed)

    pl.seed_everything(seed)

    root_dir = os.path.join(hparams.root_dir, hparams.dataset)
    datamodule = DataModule(
        root_dir=root_dir,
        spurious_label=hparams.spurious_label,
        stratified_sampling=hparams.stratified_sampling,
        transforms=None,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=use_cuda,
    )

    last_ckpt_dir = os.path.join(hparams.save_dir, experiment_name, "last.ckpt")
    if os.path.exists(last_ckpt_dir):
        print(f"Experiment {experiment_name} already exists, resuming training")
        ckpt = torch.load(last_ckpt_dir)
    else:
        print(f"Starting new experiment {experiment_name}")
        ckpt = None

    if hparams.optimizer_name != "SGD":
        hparams.pop("momentum")
    model = ResNet(hparams)

    # callbacks
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(hparams.save_dir, experiment_name),
        filename="checkpoint-{epoch:03d}-{val/epoch-avg_acc:.3f}-{val/epoch-wga:.3f}",
        monitor="val/epoch-wga",
        save_last=True,
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
    )
    progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(hparams.save_dir, experiment_name)
    )

    accelerator = "cuda" if use_cuda else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=[0],
        benchmark=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=hparams.max_epochs,
        callbacks=[ckpt_callback, progress_bar, lr_callback],
        logger=logger,
    )

    trainer.fit(model, datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="./")

    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--root-dir", type=str, default="/home/jupyterlab/datasets")
    parser.add_argument("--spurious-label", type=str, default="Male")
    parser.add_argument("--stratified-sampling", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--optimizer-name", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    hparams = parser.parse_args()
    main(hparams)
