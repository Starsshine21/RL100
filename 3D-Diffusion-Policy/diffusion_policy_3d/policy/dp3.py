from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.multimodal_obs_encoder import MultiModalObsEncoder


def _resolve_obs_keys_from_config(
    obs_mode,
    available_keys,
    use_mask=None,
    use_right_cam_img=None,
):
    available_keys = list(available_keys)
    if obs_mode is None and use_mask is None and use_right_cam_img is None:
        return available_keys
    if obs_mode is None or (isinstance(obs_mode, str) and obs_mode.strip().lower() in {"", "auto", "none", "null"}):
        use_mask = bool(use_mask)
        use_right_cam_img = bool(use_right_cam_img)
        requested = ["right_state", "rgbm" if use_mask else "head_rgb"]
        if use_right_cam_img:
            requested.append("right_cam_img")
    elif isinstance(obs_mode, str):
        mode = obs_mode.strip().lower()
        presets = {
            "all": available_keys,
            "full": ["right_state", "rgbm", "right_cam_img"],
            "rgbm": ["right_state", "rgbm"],
            "state_rgbm": ["right_state", "rgbm"],
            "head_rgb": ["right_state", "head_rgb"],
            "head_rgb_only": ["right_state", "head_rgb"],
            "state_head_rgb": ["right_state", "head_rgb"],
            "head_rgb_right_cam": ["right_state", "head_rgb", "right_cam_img"],
            "dual_rgb": ["right_state", "head_rgb", "right_cam_img"],
            "state": ["right_state"],
            "state_only": ["right_state"],
        }
        if mode in presets:
            requested = presets[mode]
        elif "," in mode:
            requested = [key.strip() for key in mode.split(",") if key.strip()]
        else:
            requested = [obs_mode]
    else:
        requested = [str(key) for key in list(obs_mode)]

    missing = [key for key in requested if key not in available_keys]
    if missing:
        raise KeyError(
            f"Requested obs keys {missing} are not present in shape_meta.obs. "
            f"Available keys: {available_keys}"
        )
    return requested


class DP3(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            rgb_encoder_type="simple",
            rgb_encoder_cfg=None,
            # VIB parameters
            use_recon_vib=False,
            beta_recon=1.0,
            beta_kl=0.001,
            generative_mode="diffusion",
            flow_sampling_steps=None,
            obs_mode=None,
            use_mask=None,
            use_right_cam_img=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.use_recon_vib = use_recon_vib
        self.beta_recon = beta_recon
        self.beta_kl = beta_kl
        self.generative_mode = str(generative_mode).lower()
        if self.generative_mode not in {"diffusion", "flow_matching"}:
            raise ValueError(
                f"Unsupported generative_mode '{self.generative_mode}'. "
                "Expected 'diffusion' or 'flow_matching'."
            )

        # Force prediction_type to 'epsilon' for diffusion-based RL-100.
        if (
            self.generative_mode == 'diffusion'
            and noise_scheduler.config.prediction_type != 'epsilon'
        ):
            cprint(f"[DP3] Forcing prediction_type to 'epsilon' (was '{noise_scheduler.config.prediction_type}')", "yellow")
            noise_scheduler.config.prediction_type = 'epsilon'

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta_all = shape_meta['obs']
        self.obs_mode = obs_mode
        self.use_mask = None if use_mask is None else bool(use_mask)
        self.use_right_cam_img = None if use_right_cam_img is None else bool(use_right_cam_img)
        self.obs_keys = _resolve_obs_keys_from_config(
            obs_mode=obs_mode,
            available_keys=obs_shape_meta_all.keys(),
            use_mask=use_mask,
            use_right_cam_img=use_right_cam_img,
        )
        obs_shape_meta = {
            key: obs_shape_meta_all[key]
            for key in self.obs_keys
        }
        cprint(
            f"[DP3] obs_mode={self.obs_mode}, use_mask={self.use_mask}, "
            f"use_right_cam_img={self.use_right_cam_img}, active obs keys={self.obs_keys}",
            "yellow",
        )

        obs_encoder = MultiModalObsEncoder(
            observation_space=obs_shape_meta,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            use_recon_vib=use_recon_vib,
            beta_recon=beta_recon,
            beta_kl=beta_kl,
            rgb_encoder_type=rgb_encoder_type,
            rgb_encoder_cfg=rgb_encoder_cfg,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.flow_sampling_steps = (
            int(flow_sampling_steps)
            if flow_sampling_steps is not None
            else int(self.num_inference_steps)
        )


        print_params(self)

    @property
    def action_chunk_start(self) -> int:
        return self.n_obs_steps - 1

    @property
    def action_chunk_end(self) -> int:
        return self.action_chunk_start + self.n_action_steps

    @property
    def chunk_action_dim(self) -> int:
        return self.action_dim * self.n_action_steps

    @property
    def is_flow_matching_policy(self) -> bool:
        return self.generative_mode == 'flow_matching'

    def extract_action_chunk(self, action_trajectory):
        return action_trajectory[:, self.action_chunk_start:self.action_chunk_end, ...]

    def _prepare_executed_action_steps(
        self,
        executed_action_steps,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if executed_action_steps is None:
            return None
        if not torch.is_tensor(executed_action_steps):
            executed_action_steps = torch.as_tensor(executed_action_steps, device=device)
        executed_action_steps = executed_action_steps.to(device=device).long().reshape(-1)
        if executed_action_steps.numel() == 1 and batch_size != 1:
            executed_action_steps = executed_action_steps.expand(batch_size)
        if executed_action_steps.numel() != batch_size:
            raise ValueError(
                f"executed_action_steps has shape {tuple(executed_action_steps.shape)} "
                f"but batch size is {batch_size}"
            )
        return executed_action_steps.clamp(min=0, max=self.n_action_steps)

    def get_action_chunk_step_mask(
        self,
        batch_size: int,
        device: torch.device,
        executed_action_steps=None,
        dtype=None,
    ):
        executed_action_steps = self._prepare_executed_action_steps(
            executed_action_steps=executed_action_steps,
            batch_size=batch_size,
            device=device,
        )
        if executed_action_steps is None:
            return None
        step_ids = torch.arange(self.n_action_steps, device=device).view(1, -1)
        step_mask = step_ids < executed_action_steps.view(-1, 1)
        if dtype is not None:
            step_mask = step_mask.to(dtype=dtype)
        return step_mask

    def mask_action_chunk(
        self,
        action_chunk: torch.Tensor,
        executed_action_steps=None,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        step_mask = self.get_action_chunk_step_mask(
            batch_size=action_chunk.shape[0],
            device=action_chunk.device,
            executed_action_steps=executed_action_steps,
            dtype=action_chunk.dtype,
        )
        if step_mask is None:
            return action_chunk
        step_mask = step_mask.unsqueeze(-1)
        if fill_value == 0.0:
            return action_chunk * step_mask
        return action_chunk * step_mask + fill_value * (1.0 - step_mask)

    def get_action_chunk_mask(
        self,
        action_trajectory: torch.Tensor,
        executed_action_steps=None,
        dtype=None,
    ):
        action_chunk = self.extract_action_chunk(action_trajectory)
        step_mask = self.get_action_chunk_step_mask(
            batch_size=action_chunk.shape[0],
            device=action_chunk.device,
            executed_action_steps=executed_action_steps,
            dtype=dtype if dtype is not None else action_chunk.dtype,
        )
        if step_mask is None:
            return None
        return step_mask.unsqueeze(-1).expand_as(action_chunk)

    def flatten_action_chunk(self, action_trajectory, executed_action_steps=None, fill_value: float = 0.0):
        action_chunk = self.extract_action_chunk(action_trajectory)
        action_chunk = self.mask_action_chunk(
            action_chunk,
            executed_action_steps=executed_action_steps,
            fill_value=fill_value,
        )
        return action_chunk.reshape(action_chunk.shape[0], -1)

    def _sample_action_from_global_cond(self, global_cond, initial_noise=None):
        batch_size = global_cond.shape[0]
        device = global_cond.device
        dtype = next(self.parameters()).dtype
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            initial_noise=initial_noise,
            **self.kwargs,
        )
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action = self.extract_action_chunk(action_pred)
        return {
            'action': action,
            'action_pred': action_pred,
            'naction_pred': naction_pred,
        }

    def predict_action_from_global_cond(
        self,
        global_cond: torch.Tensor,
        initial_noise: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        return self._sample_action_from_global_cond(global_cond, initial_noise=initial_noise)

    def encode_obs_global_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).shape[0]
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        if "cross_attention" in self.condition_type:
            return nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        return nobs_features.reshape(batch_size, -1)

    def _get_empty_action_condition(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        return cond_data, cond_mask

    def _flow_integration_grid(self, device: torch.device, dtype: torch.dtype):
        flow_steps = max(int(self.flow_sampling_steps), 1)
        h = 1.0 / float(flow_steps)
        ts = torch.arange(flow_steps, device=device, dtype=dtype) / float(flow_steps)
        return ts, h

    def evaluate_flow_vector_field(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        local_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            sample=sample,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
        )

    def _conditional_flow_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        initial_noise=None,
    ):
        if "cross_attention" in self.condition_type:
            raise NotImplementedError("Flow-matching sampling does not support cross_attention conditioning.")

        if initial_noise is not None:
            trajectory = initial_noise.to(
                device=condition_data.device,
                dtype=condition_data.dtype,
            ).clone()
        else:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )

        ts, h = self._flow_integration_grid(
            device=condition_data.device,
            dtype=condition_data.dtype,
        )

        for t in ts:
            trajectory[condition_mask] = condition_data[condition_mask]
            t_batch = t.expand(trajectory.shape[0])
            velocity = self.evaluate_flow_vector_field(
                sample=trajectory,
                timestep=t_batch,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = trajectory + h * velocity

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def sample_qam_sde_trajectory_from_global_cond(
        self,
        global_cond: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        flow_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_flow_matching_policy:
            raise RuntimeError("QAM trajectory sampling requires generative_mode='flow_matching'.")
        if not self.obs_as_global_cond:
            raise NotImplementedError("QAM integration currently requires obs_as_global_cond=True.")
        if "cross_attention" in self.condition_type:
            raise NotImplementedError("QAM integration currently requires film-style conditioning.")

        batch_size = global_cond.shape[0]
        device = global_cond.device
        dtype = global_cond.dtype
        flow_steps = int(flow_steps) if flow_steps is not None else int(self.flow_sampling_steps)
        flow_steps = max(flow_steps, 1)
        h = 1.0 / float(flow_steps)

        cond_data, cond_mask = self._get_empty_action_condition(batch_size, device, dtype)
        if generator is None:
            x = torch.randn_like(cond_data)
        else:
            x = torch.randn(
                cond_data.shape,
                generator=generator,
                dtype=dtype,
            ).to(device)

        xs = []
        ts = []
        with torch.no_grad():
            for step_idx in range(flow_steps):
                t_scalar = float(step_idx) / float(flow_steps)
                t_batch = torch.full((batch_size,), t_scalar, device=device, dtype=dtype)

                x[cond_mask] = cond_data[cond_mask]
                x_current = x.clone()
                xs.append(x_current)
                ts.append(t_batch)

                velocity = self.evaluate_flow_vector_field(
                    sample=x_current,
                    timestep=t_batch,
                    global_cond=global_cond,
                )

                if step_idx == flow_steps - 1:
                    x = x + h * velocity
                else:
                    denom = t_scalar + h
                    sigma = (2.0 * (1.0 - t_scalar + h) / denom) ** 0.5
                    if generator is None:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.randn(
                            x.shape,
                            generator=generator,
                            dtype=dtype,
                        ).to(device)
                    x = x + h * (2.0 * velocity - x / denom) + (h ** 0.5) * sigma * noise
                x[cond_mask] = cond_data[cond_mask]

        return torch.stack(xs, dim=0), torch.stack(ts, dim=0), x.detach()
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            initial_noise=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        if self.is_flow_matching_policy:
            return self._conditional_flow_sample(
                condition_data=condition_data,
                condition_mask=condition_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                generator=generator,
                initial_noise=initial_noise,
            )

        model = self.model
        scheduler = self.noise_scheduler

        if initial_noise is not None:
            trajectory = initial_noise.to(
                device=condition_data.device,
                dtype=condition_data.dtype,
            ).clone()
        else:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        if self.obs_as_global_cond:
            return self._sample_action_from_global_cond(global_cond)

        # run sampling for inpainting-based conditioning
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action = self.extract_action_chunk(action_pred)

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _compute_flow_matching_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )

            if self.use_recon_vib:
                nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
            else:
                nobs_features = self.obs_encoder(this_nobs)
                reg_loss_dict = {
                    'kl_loss': 0.0,
                    'recon_loss': 0.0,
                    'recon_loss_pc': 0.0,
                    'recon_loss_state': 0.0,
                    'total_reg_loss': 0.0
                }

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            if self.use_recon_vib:
                nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
            else:
                nobs_features = self.obs_encoder(this_nobs)
                reg_loss_dict = {
                    'kl_loss': 0.0,
                    'recon_loss': 0.0,
                    'recon_loss_pc': 0.0,
                    'recon_loss_state': 0.0,
                    'total_reg_loss': 0.0
                }

            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        x0 = torch.randn_like(trajectory)
        t = torch.rand((batch_size,), device=trajectory.device, dtype=trajectory.dtype)
        t_view = t.view(batch_size, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * trajectory
        target_velocity = trajectory - x0

        loss_mask = ~condition_mask
        x_t[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            sample=x_t,
            timestep=t,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        loss = F.mse_loss(pred, target_velocity, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        total_reg_loss = reg_loss_dict.get('total_reg_loss', 0.0)
        if isinstance(total_reg_loss, torch.Tensor):
            loss = loss + total_reg_loss
            total_reg_loss_value = float(total_reg_loss.item())
        else:
            total_reg_loss_value = float(total_reg_loss)

        loss_dict = {
            'bc_loss': float(loss.item()),
            'fm_loss': float((loss.item() - total_reg_loss_value)),
            'kl_loss': float(reg_loss_dict.get('kl_loss', 0.0)),
            'recon_loss': float(reg_loss_dict.get('recon_loss', 0.0)),
            'recon_loss_pc': float(reg_loss_dict.get('recon_loss_pc', 0.0)),
            'recon_loss_state': float(reg_loss_dict.get('recon_loss_state', 0.0)),
            'total_reg_loss': total_reg_loss_value,
        }
        return loss, loss_dict

    def compute_loss(self, batch):
        if self.is_flow_matching_policy:
            return self._compute_flow_matching_loss(batch)

        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

            # Extract features with VIB loss if enabled
            if self.use_recon_vib:
                nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
            else:
                nobs_features = self.obs_encoder(this_nobs)
                reg_loss_dict = {
                    'kl_loss': 0.0,
                    'recon_loss': 0.0,
                    'recon_loss_pc': 0.0,
                    'recon_loss_state': 0.0,
                    'total_reg_loss': 0.0
                }

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            # Extract features with VIB loss if enabled
            if self.use_recon_vib:
                nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
            else:
                nobs_features = self.obs_encoder(this_nobs)
                reg_loss_dict = {
                    'kl_loss': 0.0,
                    'recon_loss': 0.0,
                    'recon_loss_pc': 0.0,
                    'recon_loss_state': 0.0,
                    'total_reg_loss': 0.0
                }

            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # Add VIB regularization loss
        reg_loss = reg_loss_dict.get('total_reg_loss', 0.0)
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
            reg_loss = reg_loss.mean()

        total_loss = loss + reg_loss

        loss_dict = {
            'bc_loss': loss.item(),
            'kl_loss': reg_loss_dict.get('kl_loss', 0.0),
            'recon_loss': reg_loss_dict.get('recon_loss', 0.0),
            'recon_loss_pc': reg_loss_dict.get('recon_loss_pc', 0.0),
            'recon_loss_state': reg_loss_dict.get('recon_loss_state', 0.0),
            'total_reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")

        return total_loss, loss_dict
