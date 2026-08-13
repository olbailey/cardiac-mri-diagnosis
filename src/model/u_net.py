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
        """U-Net for classifying images. Input image must be have matching height and Width, 
        and be divisible by 16 given this model has 4 layers (2^4)

        Args:
            num_classes int: Number of classes the model needs to segment.
            in_channels (int, optional): Number of channels the image has, e.g. RGB has 3. Defaults to 1.
            base_c (int, optional): _description_. Defaults to 64.
        """
        super().__init__()

        # Encoder
        self.encoders: list[DoubleConv] = [DoubleConv(in_channels, base_c)]
        for i in range(0, encoder_depth - 1):
            self.encoders.append(DoubleConv(base_c * int(math.pow(2, i)), base_c * int(math.pow(2, i+1))))
        # self.enc1 = DoubleConv(in_channels, base_c)
        # self.enc2 = DoubleConv(base_c, base_c * 2)
        # self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        # self.enc4 = DoubleConv(base_c * 4, base_c * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        # self.bottleneck = DoubleConv(base_c * 8, base_c * 16)
        self.bottleneck = DoubleConv(base_c * int(math.pow(2, encoder_depth - 1)), base_c * int(math.pow(2, encoder_depth)))

        # Decoder (transpose conv upsampling + skip connections)
        self.up_scalers: list[nn.ConvTranspose2d] = []
        self.decoders: list[DoubleConv] = []
        for i in range(encoder_depth, 0, -1):
            self.up_scalers.append(nn.ConvTranspose2d(base_c * int(math.pow(2, i)), base_c * int(math.pow(2, i-1)), kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(base_c * int(math.pow(2, i)), base_c * int(math.pow(2, i-1))))

        # self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        # self.dec4 = DoubleConv(base_c * 16, base_c * 8)   # concat doubles channels
        # self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        # self.dec3 = DoubleConv(base_c * 8, base_c * 4)
        # self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        # self.dec2 = DoubleConv(base_c * 4, base_c * 2)
        # self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        # self.dec1 = DoubleConv(base_c * 2, base_c)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        e_results = []
        for encoder in self.encoders:
            if len(e_results) == 0:
                e_results.append(self.pool(encoder(x)))
            else:
                e_results.append(self.pool(encoder(e_results[-1])))
                
        # e1 = self.pool(self.enc1(x))
        # e2 = self.pool(self.enc2(e1))
        # e3 = self.pool(self.enc3(e2))
        # e4 = self.pool(self.enc4(e3))
        b  = self.bottleneck(x)

        for i, (decoder, up) in enumerate(zip(self.decoders, self.up_scalers)):
            reversed_index = -(i+1) # grabs e_results starting from the end to the front
            b = decoder(torch.cat([up(b), e_results[reversed_index]], dim=1))
        # d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        # d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        # d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        # d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(b)

if __name__ == "__main__":
    unet = UNet(4, 4)
    print(unet)