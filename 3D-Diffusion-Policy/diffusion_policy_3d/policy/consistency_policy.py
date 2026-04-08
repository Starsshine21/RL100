from typing import Dict

import torch

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy


class ConsistencyPolicyWrapper(BasePolicy):
    """
    Runtime wrapper that exposes the 1-step consistency model through the same
    BasePolicy interface as the diffusion teacher.
    """

    def __init__(self, teacher_policy: BasePolicy, consistency_model):
        super().__init__()
        self.teacher_policy = teacher_policy
        self.consistency_model = consistency_model

    def reset(self):
        self.teacher_policy.reset()

    def set_normalizer(self, normalizer):
        self.teacher_policy.set_normalizer(normalizer)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        teacher = self.teacher_policy
        if hasattr(teacher, 'encode_obs_global_cond'):
            global_cond = teacher.encode_obs_global_cond(obs_dict)
            batch_size = global_cond.shape[0]
        else:
            nobs = teacher.normalizer.normalize(obs_dict)
            if not teacher.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]

            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :teacher.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = teacher.obs_encoder(this_nobs)
            batch_size = next(iter(obs_dict.values())).shape[0]
            if "cross_attention" in getattr(teacher, 'condition_type', ''):
                global_cond = nobs_features.reshape(batch_size, teacher.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)

        naction_pred = self.consistency_model.predict_action(
            batch_size=batch_size,
            horizon=teacher.horizon,
            global_cond=global_cond,
            device=global_cond.device,
        )
        action_pred = teacher.normalizer['action'].unnormalize(naction_pred)
        action = teacher.extract_action_chunk(action_pred)

        return {
            'action': action,
            'action_pred': action_pred,
        }
