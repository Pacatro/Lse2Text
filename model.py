import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, hidden_dims: list = [64, 128, 256, 512], num_classes: int = 10):
        super(VGG, self).__init__()
        layers = []
        in_channels = 3

        for dim in hidden_dims:
            layers += [
                nn.Conv2d(
                    in_channels=in_channels, out_channels=dim, kernel_size=3, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = dim

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
