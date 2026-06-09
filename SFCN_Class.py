import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCN(nn.Module):
    """
    Simple Fully Convolutional Network (SFCN) for 3-D volumetric classification.

    Five convolutional blocks (Conv3d → InstanceNorm3d → ReLU → MaxPool3d →
    Dropout3d) progressively increase the number of feature maps while halving
    the spatial resolution. A 1×1 bottleneck convolution projects to 64 channels,
    followed by a per-voxel classifier conv, global average pooling, and dropout
    to produce final class logits.

    Used as the single-tracer baseline classifier (FDG and AMY models).
    For the unified model, see SFCNEncoder (SFCN_Encoder.py) which replaces
    the classification head with a fixed-size feature embedding.

    Args:
        in_channels : Number of input channels (default: 1 for single-channel PET).
        num_classes : Number of output classes (default: 2 for CN/AD).
        norm_layer  : Normalisation layer constructor (default: InstanceNorm3d).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 norm_layer=nn.InstanceNorm3d):
        super().__init__()

        # Block 1: 1 → 32 channels
        self.conv1 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.norm1 = norm_layer(32)
        self.pool1 = nn.MaxPool3d(2)

        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.norm2 = norm_layer(64)
        self.pool2 = nn.MaxPool3d(2)

        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.norm3 = norm_layer(128)
        self.pool3 = nn.MaxPool3d(2)

        # Block 4: 128 → 256 channels
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.norm4 = norm_layer(256)
        self.pool4 = nn.MaxPool3d(2)

        # Block 5: 256 → 256 channels
        self.conv5 = nn.Conv3d(256, 256, 3, padding=1)
        self.norm5 = norm_layer(256)
        self.pool5 = nn.MaxPool3d(2)

        # Bottleneck: 256 → 64 channels (1×1 conv)
        self.conv6      = nn.Conv3d(256, 64, 1)
        self.norm6      = norm_layer(64)

        # Per-voxel classifier → global average pool → logits
        self.classifier = nn.Conv3d(64, num_classes, 1)
        self.gap        = nn.AdaptiveAvgPool3d(1)
        self.flatten    = nn.Flatten()

        self.drop_block = nn.Dropout3d(0.5)
        self.drop_head  = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, X, Y, Z) — normalised 3-D PET volume.

        Returns:
            logits : (B, num_classes)
        """
        x = F.relu(self.norm1(self.conv1(x))); x = self.pool1(x); x = self.drop_block(x)
        x = F.relu(self.norm2(self.conv2(x))); x = self.pool2(x); x = self.drop_block(x)
        x = F.relu(self.norm3(self.conv3(x))); x = self.pool3(x); x = self.drop_block(x)
        x = F.relu(self.norm4(self.conv4(x))); x = self.pool4(x); x = self.drop_block(x)
        x = F.relu(self.norm5(self.conv5(x))); x = self.pool5(x)

        x = F.relu(self.norm6(self.conv6(x)))
        x = self.classifier(x)     # (B, num_classes, d, h, w)
        x = self.gap(x)            # (B, num_classes, 1, 1, 1)
        x = self.drop_head(x)
        x = self.flatten(x)        # (B, num_classes)
        return x
