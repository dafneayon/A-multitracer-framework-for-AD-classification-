"""
utils.py
--------
Model factory for all architectures used in this project.

The central function `get_model` returns a ready-to-use PyTorch model
given a name string and configuration parameters. Supported models:

  densenet   : DenseNet-121 (MONAI)
  resnet     : ResNet-10 (MONAI)
  efficientnet: EfficientNet-B0 (MONAI)
  senet      : SE-Net (MONAI)
  vit        : Vision Transformer (MONAI) — requires config_file.py for image size
  sfcn       : Simple Fully Convolutional Network (custom, see SFCN_Class.py)
               When feature_mode=True, returns SFCNEncoder (see SFCN_Encoder.py)
               for use in the unified model's feature-fusion pipeline.
"""

import torch.nn as nn
from monai.networks.nets import DenseNet121, EfficientNetBN, ViT, SENet, resnet10

try:
    from SFCN_Class import SFCN
    HAS_SFCN = True
except Exception:
    HAS_SFCN = False

try:
    from SFCN_Encoder import SFCNEncoder
    HAS_SFCN_ENCODER = True
except Exception:
    HAS_SFCN_ENCODER = False


def get_model(
    model_name: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    norm_type: str = "instance",
    dropout: float = 0.5,
    feature_mode: bool = False,
    feature_dim: int = 64,
):
    """
    Instantiate a model by name.

    Args:
        model_name   : One of 'densenet', 'resnet', 'efficientnet', 'senet',
                       'vit', 'sfcn'.
        spatial_dims : Number of spatial dimensions (default: 3 for 3-D PET).
        in_channels  : Number of input channels (default: 1).
        out_channels : Number of output classes (default: 2 for CN/AD).
        norm_type    : Normalisation type — 'batch' or 'instance' (default).
                       Applied to SFCN only; other models use their defaults.
        dropout      : Dropout probability (default: 0.5).
        feature_mode : If True and model_name='sfcn', returns SFCNEncoder
                       (outputs a feature embedding) instead of the full
                       classifier. Used by the unified model. (default: False)
        feature_dim  : Output embedding size when feature_mode=True (default: 64).

    Returns:
        model : Uninitialised PyTorch nn.Module.

    Raises:
        ValueError  : If model_name is not recognised.
        ImportError : If SFCN_Class.py or SFCN_Encoder.py is not found.
    """
    model_name = model_name.lower()
    norm_type  = norm_type.lower()

    if model_name == "densenet":
        return DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout,
        )

    elif model_name == "resnet":
        return resnet10(
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=out_channels,
        )

    elif model_name == "efficientnet":
        return EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=out_channels,
            dropout_rate=dropout,
        )

    elif model_name == "senet":
        return SENet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            block="se_bottleneck",
            layers=[3, 4, 6, 3],
            groups=64,
            reduction=16,
            num_classes=out_channels,
            dropout_prob=dropout,
        )

    elif model_name == "vit":
        from config_file import params as _P
        return ViT(
            spatial_dims=spatial_dims,
            img_size=(_P["imagex"], _P["imagey"], _P["imagez"]),
            in_channels=in_channels,
            num_classes=out_channels,
            pos_embed="conv",
            patch_size=(16, 16, 16),
            dropout_rate=dropout,
        )

    elif model_name == "sfcn":
        norm_map = {"batch": nn.BatchNorm3d, "instance": nn.InstanceNorm3d}
        Norm     = norm_map.get(norm_type, nn.InstanceNorm3d)

        if feature_mode:
            if not HAS_SFCN_ENCODER:
                raise ImportError(
                    "SFCNEncoder not found. Make sure SFCN_Encoder.py is in the "
                    "same directory."
                )
            return SFCNEncoder(
                in_channels=in_channels,
                feature_dim=feature_dim,
                norm_layer=Norm,
                p_drop_block=min(dropout, 0.5),
                p_drop_head=min(dropout, 0.5),
            )
        else:
            if not HAS_SFCN:
                raise ImportError(
                    "SFCN not found. Make sure SFCN_Class.py is in the same "
                    "directory."
                )
            return SFCN(
                in_channels=in_channels,
                num_classes=out_channels,
                norm_layer=Norm,
            )

    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. "
            "Choose from: densenet | resnet | efficientnet | senet | vit | sfcn"
        )
