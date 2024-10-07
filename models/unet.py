import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool1d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool1d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool1d(enc3, kernel_size=2, stride=2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool1d(enc4, kernel_size=2, stride=2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, F.interpolate(enc4, size=dec4.shape[2])), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, F.interpolate(enc3, size=dec3.shape[2])), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, F.interpolate(enc2, size=dec2.shape[2])), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, F.interpolate(enc1, size=dec1.shape[2])), dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.final_conv(dec1)
        output = F.interpolate(output, size=x.shape[2], mode='linear', align_corners=False)
        return torch.sigmoid(output)