from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and (num_channels % num_groups) != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class SpatialCropper(nn.Module):
    def __init__(self, crop_shape: Optional[Sequence[int]] = None):
        super().__init__()
        self.crop_shape = None if crop_shape is None else tuple(int(x) for x in crop_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.crop_shape is None:
            return x

        crop_h, crop_w = self.crop_shape
        _, _, height, width = x.shape
        if crop_h > height or crop_w > width:
            raise ValueError(
                f"Crop shape {self.crop_shape} exceeds input spatial size {(height, width)}"
            )
        if crop_h == height and crop_w == width:
            return x

        if self.training:
            top = torch.randint(
                low=0,
                high=height - crop_h + 1,
                size=(1,),
                device=x.device,
            ).item()
            left = torch.randint(
                low=0,
                high=width - crop_w + 1,
                size=(1,),
                device=x.device,
            ).item()
        else:
            top = max((height - crop_h) // 2, 0)
            left = max((width - crop_w) // 2, 0)

        return x[:, :, top:top + crop_h, left:left + crop_w]


class BasicResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_group_norm: bool = True,
    ):
        super().__init__()

        norm1 = _make_group_norm(out_channels) if use_group_norm else nn.BatchNorm2d(out_channels)
        norm2 = _make_group_norm(out_channels) if use_group_norm else nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = norm1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = norm2
        self.activation = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            proj_norm = (
                _make_group_norm(out_channels)
                if use_group_norm else nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                proj_norm,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.activation(x)
        return x


class DPStyleRGBEncoder(nn.Module):
    """
    Lightweight ResNet-style image encoder inspired by Diffusion Policy's
    robomimic-backed observation encoder, but implemented without external
    robomimic / torchvision dependencies.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        crop_shape: Optional[Sequence[int]] = None,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
        use_group_norm: bool = True,
    ):
        super().__init__()

        stage_channels = tuple(int(x) for x in stage_channels)
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must contain at least one stage.")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1.")

        first_channels = stage_channels[0]
        stem_norm = (
            _make_group_norm(first_channels)
            if use_group_norm else nn.BatchNorm2d(first_channels)
        )
        self.cropper = SpatialCropper(crop_shape=crop_shape)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                first_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            stem_norm,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        prev_channels = first_channels
        for stage_idx, channels in enumerate(stage_channels):
            blocks = []
            stride = 1 if stage_idx == 0 else 2
            blocks.append(
                BasicResidualBlock(
                    in_channels=prev_channels,
                    out_channels=channels,
                    stride=stride,
                    use_group_norm=use_group_norm,
                )
            )
            for _ in range(blocks_per_stage - 1):
                blocks.append(
                    BasicResidualBlock(
                        in_channels=channels,
                        out_channels=channels,
                        stride=1,
                        use_group_norm=use_group_norm,
                    )
                )
            stages.append(nn.Sequential(*blocks))
            prev_channels = channels
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(prev_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cropper(x)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
