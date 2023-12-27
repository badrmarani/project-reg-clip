import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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


class CelebA(Dataset):
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


if __name__ == "__main__":
    ds = CelebA(
        root_dir="/home/jupyterlab/datasets/celeba",
        spurious_label="Male",
        split=2,
        transforms=None,
    )
    print(len(ds))
    x = ds[0]
    for k, v in x.items():
        print(k, v.size() if isinstance(v, torch.Tensor) else v)
