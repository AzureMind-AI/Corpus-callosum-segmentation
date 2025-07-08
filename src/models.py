import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualBlock(nn.Module):
    """Residual Block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AttentionBlock(nn.Module):
    """Attention mechanism for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.bn_final(self.conv_final(x)))
        return self.dropout(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (Expansive Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # Final output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
  
    def forward(self, x: torch.Tensor, return_latent: bool = False):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)   
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck (Latent Space)
        latent_space = self.bottleneck(x4)
        if return_latent:
            return latent_space

        # Decoder with skip connections
        up1_out = self.up1(latent_space)
        x = torch.cat([x4, up1_out], dim=1)
        x = self.conv_up1(x)

        up2_out = self.up2(x)
        x = torch.cat([x3, up2_out], dim=1)
        x = self.conv_up2(x)

        up3_out = self.up3(x)
        x = torch.cat([x2, up3_out], dim=1)
        x = self.conv_up3(x)

        up4_out = self.up4(x)
        x = torch.cat([x1, up4_out], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits


class UNet_Enhanced(nn.Module):
    """Enhanced UNet with attention, residual connections, and ASPP for better performance"""
    def __init__(self, in_channels: int, n_classes: int, base_channels: int = 64):
        super(UNet_Enhanced, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Encoder with residual connections
        self.inc = ResidualBlock(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 4, base_channels * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 8, base_channels * 16))
        
        # Bottleneck with ASPP
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ASPP(base_channels * 16, base_channels * 32)
        )
        
        # Attention blocks
        self.att1 = AttentionBlock(F_g=base_channels * 32, F_l=base_channels * 16, F_int=base_channels * 8)
        self.att2 = AttentionBlock(F_g=base_channels * 16, F_l=base_channels * 8, F_int=base_channels * 4)
        self.att3 = AttentionBlock(F_g=base_channels * 8, F_l=base_channels * 4, F_int=base_channels * 2)
        self.att4 = AttentionBlock(F_g=base_channels * 4, F_l=base_channels * 2, F_int=base_channels)
        self.att5 = AttentionBlock(F_g=base_channels * 2, F_l=base_channels, F_int=base_channels // 2)
        
        # Decoder with residual connections
        self.up1 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, kernel_size=2, stride=2)
        self.conv_up1 = ResidualBlock(base_channels * 32, base_channels * 16)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv_up2 = ResidualBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up3 = ResidualBlock(base_channels * 8, base_channels * 4)
        
        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up4 = ResidualBlock(base_channels * 4, base_channels * 2)
        
        self.up5 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up5 = ResidualBlock(base_channels * 2, base_channels)
        
        # Deep supervision outputs
        self.deep_sup1 = nn.Conv2d(base_channels * 16, n_classes, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(base_channels * 8, n_classes, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(base_channels * 4, n_classes, kernel_size=1)
        
        # Final output layer with dropout
        self.dropout = nn.Dropout2d(0.1)
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor, return_latent: bool = False, deep_supervision: bool = False):
        # Encoder
        x1 = self.inc(x)          # 64
        x2 = self.down1(x1)       # 128
        x3 = self.down2(x2)       # 256
        x4 = self.down3(x3)       # 512
        x5 = self.down4(x4)       # 1024
        
        # Bottleneck
        latent_space = self.bottleneck(x5)  # 2048
        if return_latent:
            return latent_space
        
        # Decoder with attention
        up1_out = self.up1(latent_space)
        x5_att = self.att1(g=up1_out, x=x5)
        d1 = torch.cat([x5_att, up1_out], dim=1)
        d1 = self.conv_up1(d1)
        
        up2_out = self.up2(d1)
        x4_att = self.att2(g=up2_out, x=x4)
        d2 = torch.cat([x4_att, up2_out], dim=1)
        d2 = self.conv_up2(d2)
        
        up3_out = self.up3(d2)
        x3_att = self.att3(g=up3_out, x=x3)
        d3 = torch.cat([x3_att, up3_out], dim=1)
        d3 = self.conv_up3(d3)
        
        up4_out = self.up4(d3)
        x2_att = self.att4(g=up4_out, x=x2)
        d4 = torch.cat([x2_att, up4_out], dim=1)
        d4 = self.conv_up4(d4)
        
        up5_out = self.up5(d4)
        x1_att = self.att5(g=up5_out, x=x1)
        d5 = torch.cat([x1_att, up5_out], dim=1)
        d5 = self.conv_up5(d5)
        
        # Apply dropout before final layer
        d5 = self.dropout(d5)
        logits = self.outc(d5)
        
        if deep_supervision and self.training:
            # Deep supervision outputs
            ds1 = self.deep_sup1(d1)
            ds2 = self.deep_sup2(d2)
            ds3 = self.deep_sup3(d3)
            
            # Upsample deep supervision outputs to match input size
            ds1 = F.interpolate(ds1, size=x.shape[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=True)
            ds3 = F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            return logits, ds1, ds2, ds3
        
        return logits