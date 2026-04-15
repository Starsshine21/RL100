import copy
from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from diffusion_policy_3d.model.vision.pointnet_extractor import (
    PointNetEncoderXYZ,
    PointNetEncoderXYZRGB,
    create_mlp,
    chamfer_distance_fallback,
    pytorch3d_chamfer_distance,
)
from diffusion_policy_3d.model.vision.dp_image_encoder import DPStyleRGBEncoder


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _spec_has_shape(spec: Union[dict, tuple, list]) -> bool:
    if isinstance(spec, dict):
        return "shape" in spec
    if isinstance(spec, (tuple, list)):
        return False
    if hasattr(spec, "get"):
        try:
            return spec.get("shape", None) is not None
        except Exception:
            return False
    return False


def _infer_obs_type(key: str, spec: Union[dict, tuple, list]) -> str:
    if _spec_has_shape(spec):
        obs_type = str(spec.get("type", "")).strip().lower()
        if obs_type:
            return obs_type
        shape = tuple(spec["shape"])
    else:
        shape = tuple(spec)

    if len(shape) == 4:
        return "rgb"
    if len(shape) == 2:
        return "point_cloud"
    return "low_dim"


def _extract_shape(spec: Union[dict, tuple, list]) -> Tuple[int, ...]:
    if _spec_has_shape(spec):
        return tuple(spec["shape"])
    return tuple(spec)


def _is_low_dim_type(obs_type: str) -> bool:
    return obs_type.startswith("low_dim")


def _image_shape_is_chw(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 3:
        raise ValueError(f"RGB observation must be rank-3, got shape={shape}")
    c_first = shape[0] <= 4 and shape[-1] > 4
    c_last = shape[-1] <= 4 and shape[0] > 4
    if c_first:
        return True
    if c_last:
        return False
    return shape[0] <= 4


class SimpleRGBEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_dims = [32, 64, 128, 256]
        layers = []
        prev_channels = in_channels
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            prev_channels = hidden_dim
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(prev_channels, out_channels),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalObsEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        use_recon_vib=False,
        beta_recon=1.0,
        beta_kl=0.001,
        rgb_encoder_output_dim=None,
        rgb_encoder_type="simple",
        rgb_encoder_cfg=None,
    ):
        super().__init__()
        self.use_pc_color = bool(use_pc_color)
        self.pointnet_type = str(pointnet_type)
        self.use_recon_vib = bool(use_recon_vib)
        self.beta_recon = float(beta_recon)
        self.beta_kl = float(beta_kl)
        self.chamfer_fallback_chunk_size = 32
        self._chamfer_backend_logged = False
        self.img_crop_shape = (
            None if img_crop_shape is None
            else tuple(int(x) for x in img_crop_shape)
        )
        self.rgb_encoder_output_dim = int(
            rgb_encoder_output_dim if rgb_encoder_output_dim is not None else out_channel
        )
        self.rgb_encoder_type = str(rgb_encoder_type).lower()
        self.rgb_encoder_cfg = copy.deepcopy(rgb_encoder_cfg)

        self.obs_specs = OrderedDict()
        self.low_dim_keys = []
        self.point_cloud_keys = []
        self.rgb_keys = []

        for key, spec in observation_space.items():
            obs_type = _infer_obs_type(key, spec)
            shape = _extract_shape(spec)
            self.obs_specs[key] = {
                "shape": shape,
                "type": obs_type,
            }
            if obs_type == "point_cloud":
                self.point_cloud_keys.append(key)
            elif obs_type == "rgb":
                self.rgb_keys.append(key)
            elif _is_low_dim_type(obs_type):
                self.low_dim_keys.append(key)
            else:
                raise NotImplementedError(f"Unsupported obs type '{obs_type}' for key '{key}'")

        self.primary_low_dim_key = self.low_dim_keys[0] if self.low_dim_keys else None
        self.primary_point_cloud_key = self.point_cloud_keys[0] if self.point_cloud_keys else None

        if self.use_recon_vib:
            if len(self.point_cloud_keys) != 1:
                raise NotImplementedError(
                    "Recon/VIB currently supports exactly one point-cloud observation."
                )
            if len(self.rgb_keys) > 0:
                raise NotImplementedError(
                    "Recon/VIB with RGB observations is not supported in this implementation."
                )

        cprint(f"[MultiModalObsEncoder] obs keys: {list(self.obs_specs.keys())}", "yellow")
        cprint(f"[MultiModalObsEncoder] point clouds: {self.point_cloud_keys}", "yellow")
        cprint(f"[MultiModalObsEncoder] rgb: {self.rgb_keys}", "yellow")
        cprint(f"[MultiModalObsEncoder] low-dim: {self.low_dim_keys}", "yellow")

        self.pointcloud_extractors = nn.ModuleDict()
        self.rgb_extractors = nn.ModuleDict()
        self.low_dim_extractors = nn.ModuleDict()
        self.low_dim_recon_decoders = nn.ModuleDict()

        self.n_output_channels = 0

        if self.pointnet_type != "pointnet":
            raise NotImplementedError(f"pointnet_type: {self.pointnet_type}")

        for key in self.point_cloud_keys:
            spec = self.obs_specs[key]
            pc_cfg = copy.deepcopy(pointcloud_encoder_cfg)
            if pc_cfg is None:
                raise ValueError("pointcloud_encoder_cfg is required for point-cloud observations.")

            effective_channels = 6 if self.use_pc_color else 3
            pc_cfg.in_channels = effective_channels
            pc_cfg.use_vib = self.use_recon_vib and key == self.primary_point_cloud_key
            pointcloud_output_dim = int(getattr(pc_cfg, "out_channels", out_channel))
            if effective_channels > 3:
                self.pointcloud_extractors[key] = PointNetEncoderXYZRGB(**pc_cfg)
            else:
                self.pointcloud_extractors[key] = PointNetEncoderXYZ(**pc_cfg)
            self.n_output_channels += pointcloud_output_dim
            cprint(
                f"[MultiModalObsEncoder] point-cloud key '{key}' shape={spec['shape']} -> {pointcloud_output_dim}",
                "cyan",
            )

        if len(state_mlp_size) == 0:
            raise RuntimeError("State MLP size is empty.")
        net_arch = [] if len(state_mlp_size) == 1 else list(state_mlp_size[:-1])
        low_dim_output_dim = int(state_mlp_size[-1])
        for key in self.low_dim_keys:
            input_dim = int(np.prod(self.obs_specs[key]["shape"]))
            self.low_dim_extractors[key] = nn.Sequential(
                *create_mlp(input_dim, low_dim_output_dim, net_arch, state_mlp_activation_fn)
            )
            if self.use_recon_vib and key == self.primary_low_dim_key:
                self.low_dim_recon_decoders[key] = nn.Sequential(
                    nn.Linear(low_dim_output_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim),
                )
            self.n_output_channels += low_dim_output_dim
            cprint(
                f"[MultiModalObsEncoder] low-dim key '{key}' shape={self.obs_specs[key]['shape']} -> {low_dim_output_dim}",
                "cyan",
            )

        for key in self.rgb_keys:
            img_shape = self.obs_specs[key]["shape"]
            in_channels = img_shape[0] if _image_shape_is_chw(img_shape) else img_shape[-1]
            if self.rgb_encoder_type == "simple":
                self.rgb_extractors[key] = SimpleRGBEncoder(
                    in_channels=in_channels,
                    out_channels=self.rgb_encoder_output_dim,
                )
            elif self.rgb_encoder_type in {"dp", "dp_resnet", "dp_style"}:
                crop_shape = _cfg_get(self.rgb_encoder_cfg, "crop_shape", self.img_crop_shape)
                stage_channels = _cfg_get(self.rgb_encoder_cfg, "stage_channels", (32, 64, 128, 256))
                blocks_per_stage = int(_cfg_get(self.rgb_encoder_cfg, "blocks_per_stage", 2))
                use_group_norm = bool(_cfg_get(self.rgb_encoder_cfg, "use_group_norm", True))
                self.rgb_extractors[key] = DPStyleRGBEncoder(
                    in_channels=in_channels,
                    out_channels=self.rgb_encoder_output_dim,
                    crop_shape=crop_shape,
                    stage_channels=stage_channels,
                    blocks_per_stage=blocks_per_stage,
                    use_group_norm=use_group_norm,
                )
            else:
                raise ValueError(f"Unsupported rgb_encoder_type '{self.rgb_encoder_type}'")
            self.n_output_channels += self.rgb_encoder_output_dim
            cprint(
                f"[MultiModalObsEncoder] rgb key '{key}' shape={img_shape} "
                f"encoder={self.rgb_encoder_type} -> {self.rgb_encoder_output_dim}",
                "cyan",
            )

        cprint(f"[MultiModalObsEncoder] output dim: {self.n_output_channels}", "red")

    def _prepare_point_cloud(self, key: str, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"point cloud '{key}' must be rank-3 [B, N, C], got {tuple(points.shape)}")
        target_channels = 6 if self.use_pc_color else 3
        points = points.float()
        if points.shape[-1] > target_channels:
            points = points[..., :target_channels]
        elif points.shape[-1] < target_channels:
            pad_shape = list(points.shape)
            pad_shape[-1] = target_channels - points.shape[-1]
            pad = torch.zeros(pad_shape, device=points.device, dtype=points.dtype)
            points = torch.cat([points, pad], dim=-1)
        return points

    def _prepare_low_dim(self, key: str, value: torch.Tensor) -> torch.Tensor:
        batch_size = value.shape[0]
        return value.float().reshape(batch_size, -1)

    def _prepare_rgb(self, key: str, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"rgb obs '{key}' must be rank-4 [B, H, W, C] or [B, C, H, W], got {tuple(image.shape)}")
        image = image.float()
        if _image_shape_is_chw(self.obs_specs[key]["shape"]):
            image_chw = image
        else:
            image_chw = image.permute(0, 3, 1, 2)
        if torch.isfinite(image_chw).all():
            max_val = float(image_chw.detach().amax().item()) if image_chw.numel() > 0 else 0.0
            if max_val > 1.5:
                image_chw = image_chw / 255.0
        return image_chw

    def forward(self, observations: Dict, return_reg_loss: bool = False):
        features = []
        zero_device = None

        vib_kl = None
        vib_recon_pc = None
        vib_recon_state = None

        for key in self.point_cloud_keys:
            raw_points = observations[key]
            zero_device = raw_points.device
            points = self._prepare_point_cloud(key, raw_points)
            extractor = self.pointcloud_extractors[key]
            if self.use_recon_vib and return_reg_loss and key == self.primary_point_cloud_key:
                feat, mu, logvar, recon_pc = extractor(points, return_recon=True)
                original_points_xyz = points[..., :3]
                vib_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                if pytorch3d_chamfer_distance is not None:
                    try:
                        vib_recon_pc, _ = pytorch3d_chamfer_distance(original_points_xyz, recon_pc)
                        if not self._chamfer_backend_logged:
                            cprint("[MultiModalObsEncoder] VIB recon uses pytorch3d.loss.chamfer_distance", "cyan")
                            self._chamfer_backend_logged = True
                    except Exception as exc:
                        if not self._chamfer_backend_logged:
                            cprint(
                                f"[MultiModalObsEncoder] pytorch3d Chamfer failed ({type(exc).__name__}); using torch.cdist fallback",
                                "yellow",
                            )
                            self._chamfer_backend_logged = True
                if vib_recon_pc is None:
                    vib_recon_pc = chamfer_distance_fallback(
                        original_points_xyz,
                        recon_pc,
                        chunk_size=self.chamfer_fallback_chunk_size,
                    )
            else:
                feat = extractor(points)
            features.append(feat)

        for key in self.rgb_keys:
            image = observations[key]
            zero_device = image.device
            image_chw = self._prepare_rgb(key, image)
            features.append(self.rgb_extractors[key](image_chw))

        for key in self.low_dim_keys:
            low_dim = observations[key]
            zero_device = low_dim.device
            flat = self._prepare_low_dim(key, low_dim)
            feat = self.low_dim_extractors[key](flat)
            features.append(feat)
            if self.use_recon_vib and return_reg_loss and key == self.primary_low_dim_key:
                recon = self.low_dim_recon_decoders[key](feat)
                vib_recon_state = F.mse_loss(recon, flat)

        if not features:
            raise RuntimeError("No observation features were produced.")

        final_feat = torch.cat(features, dim=-1)

        if not return_reg_loss:
            return final_feat

        device = zero_device if zero_device is not None else final_feat.device
        kl_loss = vib_kl if vib_kl is not None else torch.zeros((), device=device)
        recon_loss_pc = vib_recon_pc if vib_recon_pc is not None else torch.zeros((), device=device)
        recon_loss_state = vib_recon_state if vib_recon_state is not None else torch.zeros((), device=device)
        recon_loss = recon_loss_pc + recon_loss_state
        total_reg_loss = self.beta_recon * recon_loss + self.beta_kl * kl_loss
        reg_loss_dict = {
            "kl_loss": float(kl_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "recon_loss_pc": float(recon_loss_pc.item()),
            "recon_loss_state": float(recon_loss_state.item()),
            "total_reg_loss": total_reg_loss,
        }
        return final_feat, reg_loss_dict

    def output_shape(self):
        return self.n_output_channels
