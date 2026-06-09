import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCNEncoder(nn.Module):
    """
    SFCN feature encoder — identical convolutional backbone to SFCN_Class.py
    but without the classification head.

    Instead of producing class logits, the network projects the spatial
    feature maps to a fixed-size embedding vector of shape (B, feature_dim).
    This embedding is used by the unified model (ModalityFeatureFusion in
    saliency_maps.py) which concatenates it with a modality embedding before
    the classification head.

    Architecture:
        5 × [Conv3d → InstanceNorm3d → ReLU → MaxPool3d → Dropout3d]
        → Conv3d 1×1 (channel projection to feature_dim)
        → InstanceNorm3d → ReLU
        → AdaptiveAvgPool3d(1)
        → Dropout
        → Flatten
        → (B, feature_dim)

    Args:
        in_channels  : Number of input channels (default: 1 for single-channel PET).
        feature_dim  : Dimensionality of the output embedding (default: 64).
        norm_layer   : Normalisation layer constructor (default: InstanceNorm3d).
        p_drop_block : Dropout probability applied after each pooling block (default: 0.5).
        p_drop_head  : Dropout probability applied before the final flatten (default: 0.5).
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 64,
        norm_layer=nn.InstanceNorm3d,
        p_drop_block: float = 0.5,
        p_drop_head: float = 0.5,
    ):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.norm1 = norm_layer(32)
        self.pool1 = nn.MaxPool3d(2)

        # Block 2
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.norm2 = norm_layer(64)
        self.pool2 = nn.MaxPool3d(2)

        # Block 3
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.norm3 = norm_layer(128)
        self.pool3 = nn.MaxPool3d(2)

        # Block 4
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.norm4 = norm_layer(256)
        self.pool4 = nn.MaxPool3d(2)

        # Block 5
        self.conv5 = nn.Conv3d(256, 256, 3, padding=1)
        self.norm5 = norm_layer(256)
        self.pool5 = nn.MaxPool3d(2)

        # 1×1 projection to latent feature space
        self.conv6 = nn.Conv3d(256, feature_dim, 1)
        self.norm6 = norm_layer(feature_dim)

        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

        self.drop_block = nn.Dropout3d(p_drop_block)
        self.drop_head  = nn.Dropout(p_drop_head)

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, X, Y, Z) — normalised 3-D PET volume.

        Returns:
            features : (B, feature_dim) — dense embedding vector.
        """
        x = F.relu(self.norm1(self.conv1(x))); x = self.pool1(x); x = self.drop_block(x)
        x = F.relu(self.norm2(self.conv2(x))); x = self.pool2(x); x = self.drop_block(x)
        x = F.relu(self.norm3(self.conv3(x))); x = self.pool3(x); x = self.drop_block(x)
        x = F.relu(self.norm4(self.conv4(x))); x = self.pool4(x); x = self.drop_block(x)
        x = F.relu(self.norm5(self.conv5(x))); x = self.pool5(x)

        x = F.relu(self.norm6(self.conv6(x)))   # [B, feature_dim, d, h, w]
        x = self.gap(x)                          # [B, feature_dim, 1, 1, 1]
        x = self.drop_head(x)
        x = self.flatten(x)                      # [B, feature_dim]
        return x