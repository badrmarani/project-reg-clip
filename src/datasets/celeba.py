import os
import zipfile

import gdown
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import stratified_sampler

DTYPE = torch.int


def transform_data():
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


class CelebADataset(Dataset):
    def __init__(
        self, root_dir: str, spurious_label: str, split: int, transforms=None
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.spurious_label = spurious_label
        if transforms is None:
            self.transforms = transform_data()
        else:
            self.transforms = transforms

        self.metadata = pd.read_csv(
            os.path.join(root_dir, "list_attr_celeba.csv"),
            delim_whitespace=True,
            header=1,
            index_col=0,
        )
        for c in self.metadata.columns:
            self.metadata[c] = self.metadata[c].apply(lambda x: 0 if x == -1 else 1)

        df_partitions = pd.read_csv(
            os.path.join(root_dir, "list_eval_partition.csv"),
            delim_whitespace=True,
            header=None,
            index_col=0,
        )
        indices = df_partitions[df_partitions.iloc[:, 0] == split].index

        self.metadata = self.metadata[["Blond_Hair", spurious_label]].loc[indices, :]
        self.metadata.rename(
            columns={"Blond_Hair": "label", spurious_label: "spurious_label"},
            inplace=True,
        )
        self.metadata["group"] = self.metadata.apply(
            lambda x: 2 * x["label"] + x["spurious_label"], axis=1
        )

        self.groups = torch.as_tensor(self.metadata["group"].values, dtype=DTYPE)
        self.group_counts = (
            torch.arange(4, dtype=torch.int).unsqueeze(1).eq(self.groups).sum(dim=1)
        )
        self.labels = torch.as_tensor(self.metadata["label"].values, dtype=DTYPE)
        self.label_counts = (
            torch.arange(2, dtype=torch.int).unsqueeze(1).eq(self.labels).sum(dim=1)
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = self.metadata.iloc[index, :]

        filename = os.path.join(self.root_dir, "img_align_celeba/", sample.name)
        image = Image.open(filename).convert("RGB")
        image = self.transforms(image)

        items = {}
        items["filename"] = sample.name
        items["image"] = image
        items["label"] = torch.as_tensor(sample["label"], dtype=DTYPE)
        items["spurious_label"] = torch.as_tensor(sample["spurious_label"], dtype=DTYPE)
        items["group"] = torch.as_tensor(sample["group"], dtype=DTYPE)
        return items


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        spurious_label: str,
        stratified_sampling,
        transforms,
        batch_size,
        num_workers,
        pin_memory,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.spurious_label = spurious_label
        if transforms is None:
            self.transforms = transform_data
        else:
            self.transforms = transforms
        self.stratified_sampling = stratified_sampling

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        files = {
            "list_eval_partition.csv": "1kDqtHZHpYMe7rt1zu9pOevzUbApkDNRa",
            "list_attr_celeba.csv": "1s8CyrddcxHdvwro-_M25H7uxsDWL_1Bs",
            "img_align_celeba.zip": "1mGM-w9373aW5UJ27xa5oAsesL06JOe3h",
        }
        os.makedirs(self.root_dir, exist_ok=True)
        for file, file_id in files.items():
            path = os.path.join(self.root_dir, file)
            if not os.path.exists(path):
                print(f"Downloading {file}...")
                gdown.download(id=file_id, output=path, quiet=True)
            else:
                print(f"The file {file} already exists.")

            if file.endswith("zip") and not os.path.exists(
                os.path.join(self.root_dir, file.replace(".zip", ""))
            ):
                print(f"Unzipping {file}...")
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(self.root_dir)

    def setup(self, stage=None):
        self.train_dataset = CelebADataset(self.root_dir, self.spurious_label, 0)
        self.val_dataset = CelebADataset(self.root_dir, self.spurious_label, 1)
        self.test_dataset = CelebADataset(self.root_dir, self.spurious_label, 2)

    def train_dataloader(self):
        shuffle = False
        sampler = None
        if self.stratified_sampling:
            sampler = stratified_sampler(self.train_dataset)
        else:
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
