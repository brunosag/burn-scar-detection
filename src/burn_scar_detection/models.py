import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# ======== SMP + Multi-scale CBAM ========


class ChannelAttention(nn.Module):
    """Channel Attention Module (from Implementation A)."""

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
    """Spatial Attention Module (from Implementation A)."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_cat)
        return self.sigmoid(x_att)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (from Implementation A)."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FusionBlock(nn.Module):
    """Fuses features [f1, f2, attended_difference] (from Implementation A)."""

    def __init__(self, in_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2, attended_difference):
        fused = torch.cat([f1, f2, attended_difference], dim=1)
        return self.fusion_conv(fused)


class SmpSiameseCBAM(nn.Module):
    """Implementation A Model: Siamese U-Net with SMP backbone and multi-scale CBAM fusion."""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.model = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes,
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head

        self.cbam_blocks = nn.ModuleList(
            [CBAMBlock(ch) for ch in self.encoder.out_channels]
        )
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(ch) for ch in self.encoder.out_channels]
        )

    def forward(self, t1, t2):
        f1_features = self.encoder(t1)
        f2_features = self.encoder(t2)

        fused_skip_features = []
        for i, (f1, f2) in enumerate(zip(f1_features, f2_features)):
            difference = torch.abs(f1 - f2)
            attended_difference = self.cbam_blocks[i](difference)
            fused_output = self.fusion_blocks[i](f1, f2, attended_difference)
            fused_skip_features.append(fused_output)

        decoder_output = self.decoder(fused_skip_features)
        masks = self.segmentation_head(decoder_output)
        return masks


# ======== Custom U-Net ========


class CustomDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 (from Implementation B)."""

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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CustomCBAM(nn.Module):
    """CBAM module implementation from Implementation B."""

    def __init__(self, gate_channels, reduction_ratio=16):
        super(CustomCBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(gate_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_att = torch.sigmoid(self.channel_gate(x))
        x_channel_refined = x * channel_att
        spatial_att = self.spatial_gate(x_channel_refined)
        x_spatial_refined = x_channel_refined * spatial_att
        return x_spatial_refined


class CustomUNetSiamese(nn.Module):
    """Implementation B Model: Custom Siamese U-Net with bottleneck attention."""

    def __init__(self, n_channels, n_classes=1):
        super(CustomUNetSiamese, self).__init__()
        self.inc = CustomDoubleConv(n_channels, 64)
        self.down1_pool = nn.MaxPool2d(2)
        self.conv1 = CustomDoubleConv(64, 128)
        self.down2_pool = nn.MaxPool2d(2)
        self.conv2 = CustomDoubleConv(128, 256)

        self.cbam_bottleneck = CustomCBAM(256)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv1 = CustomDoubleConv(256 + 128 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv2 = CustomDoubleConv(128 + 64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward_encoder(self, x):
        x1 = self.inc(x)
        x2_in = self.down1_pool(x1)
        x2 = self.conv1(x2_in)
        x3_in = self.down2_pool(x2)
        x3 = self.conv2(x3_in)
        return x1, x2, x3

    def forward(self, t1, t2):
        x1_t1, x2_t1, x3_t1 = self.forward_encoder(t1)
        x1_t2, x2_t2, x3_t2 = self.forward_encoder(t2)

        x3_diff = torch.abs(x3_t1 - x3_t2)
        x3_fused = self.cbam_bottleneck(x3_diff)

        x = self.up1(x3_fused)
        x = torch.cat([x, x2_t1, x2_t2], dim=1)
        x = self.dconv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1_t1, x1_t2], dim=1)
        x = self.dconv2(x)

        logits = self.outc(x)
        return logits


# ======== Model Factory ========

MODEL_REGISTRY = {
    'smp_siamese': SmpSiameseCBAM,
    'custom_unet': CustomUNetSiamese,
}


def get_model(model_id: str, n_channels: int, n_classes: int):
    """Factory function to create the specified model architecture using a registry."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f'Unknown model architecture: {model_id}. Available: {list(MODEL_REGISTRY.keys())}'
        )

    model_class = MODEL_REGISTRY[model_id]

    print(f'Initializing {model_class.__name__} model')
    return model_class(n_channels=n_channels, n_classes=n_classes)
