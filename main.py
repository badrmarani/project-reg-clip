import json
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from src.datasets import CelebA, Nico, Waterbirds
from src.datasets.loader import get_loaders
from src.models.lightning_base import LightningBase
from src.models.resnet import ResNet


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset == "celeba":
        Dataset = CelebA
        spurious_label = "Male"
    elif args.dataset == "waterbirds":
        Dataset = Waterbirds
        spurious_label = "place"
    elif args.dataset == "nico":
        Dataset = Nico
        spurious_label = None

    root_dir = os.path.join(args.root_dir, args.dataset)
    train_dataset = Dataset(root_dir, spurious_label, 0)
    train_loader = get_loaders(
        train_dataset,
        use_stratified_sampling=args.use_stratified_sampling,
        labels=args.on_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_dataset = Dataset(root_dir, spurious_label, 1)
    val_loader = get_loaders(
        val_dataset,
        batch_size=32,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    # test_dataset = Dataset(root_dir, spurious_label, 2)
    # test_loader = get_loaders(
    #     test_dataset,
    #     batch_size=32,
    #     num_workers=args.num_workers,
    #     pin_memory=use_cuda,
    # )

    if args.use_stratified_sampling:
        msg = "_stratified_sampling"
        if args.on_labels:
            msg += "_on_labels"
        else:
            msg += "_on_groups"
    else:
        msg = ""

    experiment_name = (
        f"{args.model_name}"
        + f"_{args.dataset}"
        + f"_{spurious_label}"
        + f"_{args.weight_decay}"
        + msg
    ).lower()

    if os.path.exists(f"checkpoints/{experiment_name}/last.ckpt"):
        print(f"Experiment {experiment_name} already exists, resuming training")
        ckpt = torch.load(f"checkpoints/{experiment_name}/last.ckpt")

        if os.path.exists(f"checkpoints/{experiment_name}/configs.json"):
            print("Override config settings.")
            parser = ArgumentParser()
            args = parser.parse_args()
            with open(f"checkpoints/{experiment_name}/configs.json", "r") as f:
                args.__dict__ = json.load(f)
    else:
        print(f"Starting new experiment {experiment_name}")
        ckpt = None

    model = ResNet(args.model_name, args.num_classes, args.pretrained).to(device)
    optimizer_cfg = {"lr": args.lr, "weight_decay": args.weight_decay}
    lightning_model = LightningBase(model, optimizer_cfg).to(device)

    # callbacks
    progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)
    logger = pl.loggers.TensorBoardLogger(save_dir=f"logs/{experiment_name}/")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.save_path, f"checkpoints/{experiment_name}/"),
        filename="checkpoint-{epoch:03d}-val_acc_avg:{val/acc_avg:.5f}-val_wg_acc:{val/wg_acc:.5f}",
        monitor="val/wg_acc",
        save_last=True,
        save_top_k=3,
        mode="max",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1],
        benchmark=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback, progress_bar],
        logger=logger,
    )

    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt)

    os.makedirs(f"checkpoints/{experiment_name}/", exist_ok=True)
    with open(f"checkpoints/{experiment_name}/configs.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-path", type=str, default="./")

    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true", default=False)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=("celeba", "waterbirds", "nico"),
        default="celeba",
    )
    parser.add_argument("--root-dir", type=str, default="/home/jupyterlab/datasets")
    parser.add_argument("--use-stratified-sampling", action="store_true", default=False)
    parser.add_argument("--on-labels", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
