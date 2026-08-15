import math

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, num_classes, encoder_depth, in_channels=1, base_c=64):
        """U-Net for classifying images. Input height and width must match and be
        divisible by 2^encoder_depth, e.g. encoder_depth=4 requires size divisible by 16.

        Args:
            num_classes (int): Number of classes the model needs to segment.
            encoder_depth (int): Number of encoder/decoder stages in the model.
            in_channels (int, optional): Number of input image channels, e.g. RGB has 3. Defaults to 1.
            base_c (int, optional): Number of channels produced by the first encoder stage;
                doubles at each deeper stage. Defaults to 64.
        """
        super().__init__()

        def channels(_level):
            return int(math.pow(2, _level))

        # Encoder
        self.encoders = nn.ModuleList([DoubleConv(in_channels, base_c)])
        for level in range(0, encoder_depth - 1):
            self.encoders.append(DoubleConv(base_c * channels(level), base_c * channels(level + 1)))

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_c * channels(encoder_depth - 1), base_c * channels(encoder_depth))

        # Decoder (transpose conv upsampling + skip connections)
        self.up_scalers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(encoder_depth, 0, -1):
            self.up_scalers.append(nn.ConvTranspose2d(base_c * channels(level), base_c * channels(level - 1), kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(base_c * channels(level), base_c * channels(level - 1)))

        # Output
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        current = x
        for encoder in self.encoders:
            skip = encoder(current)
            skip_connections.append(skip)
            current = self.pool(skip)


        current  = self.bottleneck(current)

        for up_sample, decoder, skip in zip(self.up_scalers, self.decoders, reversed(skip_connections)):
            concat = torch.cat([up_sample(current), skip], dim=1)
            current = decoder(concat)

        return self.out_conv(current)

if __name__ == "__main__":
    unet = UNet(num_classes=4, encoder_depth=5)
    print(unet)