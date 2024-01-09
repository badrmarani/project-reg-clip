import torch
from timm import create_model
from torch import nn


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
    ):
        super().__init__()

        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet("resnet18", num_classes=2, pretrained=True, trainable=True)
    x = torch.randn((2, 3, 224, 224), dtype=torch.float)
    print(model(x).size())
