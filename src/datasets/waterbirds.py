import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DTYPE = torch.int


def transform_data():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


class Waterbirds(Dataset):
    def __init__(
        self, root_dir: str, spurious_label: str, split: int, transforms
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.spurious_label = spurious_label
        if transforms is None:
            self.transforms = transform_data()
        else:
            self.transforms = transforms

        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"), index_col=0)
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.metadata.rename(
            columns={"y": "label", spurious_label: "spurious_label"},
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

        filename = os.path.join(self.root_dir, sample.img_filename)
        image = Image.open(filename).convert("RGB")
        image = self.transforms(image)

        items = {}
        items["filename"] = sample.img_filename
        items["image"] = image
        items["label"] = torch.as_tensor(sample["label"], dtype=DTYPE)
        items["spurious_label"] = torch.as_tensor(sample["spurious_label"], dtype=DTYPE)
        items["group"] = torch.as_tensor(sample["group"], dtype=DTYPE)
        return items


if __name__ == "__main__":
    ds = Waterbirds(
        root_dir="/home/jupyterlab/datasets/waterbirds",
        spurious_label="place",
        split=2,
        transforms=None,
    )
    print(len(ds))
    x = ds[0]
    for k, v in x.items():
        print(k, v.size() if isinstance(v, torch.Tensor) else v)
