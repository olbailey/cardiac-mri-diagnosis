import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        return self.block(x)

class FlowersCNNClassifer(nn.Module):
    def __init__(self):
        super(FlowersCNNClassifer, self).__init__()

        self.cnn_layers = nn.Sequential( # layers still defined in sequential
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 102),
        )

    def forward(self, x):
        x = self.cnn_layers(x) # data passed through sequential layers
        x = self.classifier(x)

        return x