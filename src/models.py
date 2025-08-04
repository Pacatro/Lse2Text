import torch
from torch import nn


class CnnV1(nn.Module):
    def __init__(
        self,
        input_channel: int,
        out_channels: int,
        hidden_units: list[int] = [128, 64, 32],
        adapt_size: tuple[int, int] = (7, 7),
        p: float = 0.5,
    ):
        super(CnnV1, self).__init__()

        self.config = {
            "input_channel": input_channel,
            "out_channels": out_channels,
            "hidden_units": hidden_units,
            "adapt_size": adapt_size,
            "p": p,
        }

        layers = []
        input = input_channel

        for h in hidden_units:
            layers += [
                nn.Conv2d(
                    in_channels=input,
                    out_channels=h,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(h),
                nn.MaxPool2d(2),
            ]
            input = h

        self.features = nn.Sequential(*layers)

        self.adapt_pool = nn.AdaptiveAvgPool2d(adapt_size)
        flat_dim = hidden_units[-1] * adapt_size[0] * adapt_size[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_features=4096, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt_pool(x)
        return self.classifier(x)
