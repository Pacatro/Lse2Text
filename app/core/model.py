import torch
from torch import nn


class CnnV1(nn.Module):
    def __init__(self, config: dict):
        super(CnnV1, self).__init__()

        self.config = config

        layers = []
        input = self.config["input_channel"]

        for h in self.config["hidden_units"]:
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

        self.adapt_pool = nn.AdaptiveAvgPool2d(self.config["adapt_size"])
        flat_dim = (
            self.config["hidden_units"][-1]
            * self.config["adapt_size"][0]
            * self.config["adapt_size"][1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config["p"]),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config["p"]),
            nn.Linear(in_features=4096, out_features=self.config["out_channels"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt_pool(x)
        return self.classifier(x)
