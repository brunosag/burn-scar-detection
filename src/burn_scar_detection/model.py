import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(1, in_planes // ratio)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FusionBlock(nn.Module):
    """A block to fuse features from two encoder branches."""

    def __init__(self, in_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2):
        fused = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        return self.fusion_conv(fused)


class SiameseAttentionUNet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head

        self.cbam_blocks = nn.ModuleList([CBAM(ch) for ch in self.encoder.out_channels])

        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(ch) for ch in self.encoder.out_channels]
        )

    def forward(self, t1, t2):
        f1_features = self.encoder(t1)
        f2_features = self.encoder(t2)

        f1_attended = [cbam(f) for cbam, f in zip(self.cbam_blocks, f1_features)]
        f2_attended = [cbam(f) for cbam, f in zip(self.cbam_blocks, f2_features)]

        fused_skip_features = [
            fusion_block(f1, f2)
            for fusion_block, f1, f2 in zip(
                self.fusion_blocks, f1_attended, f2_attended
            )
        ]

        decoder_output = self.decoder(fused_skip_features)
        masks = self.segmentation_head(decoder_output)

        return masks
