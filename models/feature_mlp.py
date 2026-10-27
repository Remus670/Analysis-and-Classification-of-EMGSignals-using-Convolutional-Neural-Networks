import torch.nn as nn


class EMGFeatureMLP(nn.Module):
    def __init__(self, n_classes, n_channels=8):
        super().__init__()
        input_size = n_channels * 4
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)
