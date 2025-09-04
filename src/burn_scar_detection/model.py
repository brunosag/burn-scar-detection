import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


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
            attention_type='scse',
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(ch) for ch in self.encoder.out_channels]
        )

    def forward(self, t1, t2):
        f1_features = self.encoder(t1)
        f2_features = self.encoder(t2)

        fused_skip_features = [
            fusion_block(f1, f2)
            for fusion_block, f1, f2 in zip(
                self.fusion_blocks, f1_features, f2_features
            )
        ]

        decoder_output = self.decoder(fused_skip_features)
        masks = self.segmentation_head(decoder_output)

        return masks
