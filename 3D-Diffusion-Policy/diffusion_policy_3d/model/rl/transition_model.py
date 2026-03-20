"""
Ensemble Transition Model T_θ(s'|s, a) for RL-100
===================================================
Implements Algorithm 1 Line 6: "Train transition: T_θm(s'|s, a)"

Adapted from Uni-O4 (https://github.com/Lei-Kun/Uni-o4) ensemble dynamics,
with the following key differences for RL-100:

  * Operates in **obs-feature space** (output of DP3 obs_encoder, e.g. 256-dim)
    rather than raw observation space, because:
      - Point clouds are too high-dimensional to model directly
      - Q/V networks already consume encoded features, not raw obs
  * Provides a clean PyTorch state_dict interface for checkpoint saving
  * No external Logger dependency – prints via termcolor

Architecture:
  Input  : [obs_features (256) | norm_action (4)]  →  260-dim
  Output : Δobs_features (256) + reward (1)         →  257-dim
           (predict delta so the model only needs to
            learn the residual from the current features)
  Network: 7-member ensemble of 4-layer MLPs with Swish activations.
           Top-5 elites selected by holdout MSE.

Training (train_on_dataset):
  1. Encode entire offline dataset once with frozen policy encoder.
  2. Build (obs_feat_t, norm_action_t) → (Δfeat, reward) pairs.
  3. Gaussian NLL loss + logvar regularisation + weight decay.
  4. Early stopping on 20% holdout; save best weights per member.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from termcolor import cprint


# ---------------------------------------------------------------------------
# Building blocks (from Uni-O4 transition_model/)
# ---------------------------------------------------------------------------

class EnsembleLinear(nn.Module):
    """Batched linear layer that runs all ensemble members in parallel."""

    def __init__(self, input_dim: int, output_dim: int,
                 num_ensemble: int, weight_decay: float = 0.0) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.weight_decay = weight_decay

        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))
        nn.init.trunc_normal_(self.weight, std=1.0 / (2 * input_dim ** 0.5))

        # Snapshot weights for elite-based early stopping
        self.register_parameter(
            "saved_weight",
            nn.Parameter(self.weight.detach().clone()))
        self.register_parameter(
            "saved_bias",
            nn.Parameter(self.bias.detach().clone()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [E, B, in] or [B, in]
        if x.dim() == 2:
            x = torch.einsum('bi,eio->ebo', x, self.weight)
        else:
            x = torch.einsum('ebi,eio->ebo', x, self.weight)
        return x + self.bias

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        return self.weight_decay * 0.5 * (self.weight ** 2).sum()


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def soft_clamp(x: torch.Tensor,
               _min: Optional[torch.Tensor] = None,
               _max: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Differentiable clamping that preserves gradients."""
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class StandardScaler:
    """Mean/std normalizer for model inputs (numpy arrays)."""

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mu) / self.std

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mu = torch.as_tensor(self.mu, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mu) / std


# ---------------------------------------------------------------------------
# Ensemble Dynamics Network
# ---------------------------------------------------------------------------

class EnsembleDynamicsModel(nn.Module):
    """
    Core neural network for the ensemble transition model.

    Input  : [obs_features | action]          (obs_dim + action_dim)
    Output : mean and logvar of
             [Δobs_features | reward]         (obs_dim + 1)

    num_ensemble members run in parallel; top num_elites used for inference.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (200, 200, 200, 200),
        num_ensemble: int = 7,
        num_elites: int = 5,
        weight_decays: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        output_dim = obs_dim + 1  # Δfeatures + reward

        if weight_decays is None:
            # Default weight decays from Uni-O4 (one per layer including output)
            weight_decays = (2.5e-5,) * len(hidden_dims) + (1e-4,)
        assert len(weight_decays) == len(hidden_dims) + 1, \
            "Need one weight_decay value per layer (hidden + output)"

        dims = [obs_dim + action_dim] + list(hidden_dims)
        self.backbones = nn.ModuleList([
            EnsembleLinear(in_d, out_d, num_ensemble, wd)
            for in_d, out_d, wd in zip(dims[:-1], dims[1:], weight_decays[:-1])
        ])
        self.output_layer = EnsembleLinear(
            dims[-1], 2 * output_dim, num_ensemble, weight_decays[-1])
        self.activation = Swish()

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(output_dim) * 0.5, requires_grad=True))
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(output_dim) * -10, requires_grad=True))
        self.register_parameter(
            "elites",
            nn.Parameter(torch.arange(num_elites), requires_grad=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [E, B, input_dim] or [B, input_dim]
        for layer in self.backbones:
            x = self.activation(layer(x))
        mean, logvar = torch.chunk(self.output_layer(x), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar  # each: [E, B, output_dim]

    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)

    def set_elites(self, indexes: List[int]) -> None:
        self.register_parameter(
            'elites',
            nn.Parameter(torch.tensor(indexes), requires_grad=False))

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        loss = sum(l.get_decay_loss() for l in self.backbones)
        return loss + self.output_layer.get_decay_loss()


# ---------------------------------------------------------------------------
# High-level TransitionModel wrapper
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    World model T_θ(s'|s, a) for RL-100 (Algorithm 1, Line 6).

    Wraps EnsembleDynamicsModel with:
      - Dataset construction (encode obs with policy encoder)
      - Elite-based early stopping training loop
      - predict_next_features() for use in IQL critic training

    Args:
        obs_feature_dim : dimension of flattened obs encoder output
                          (= policy.obs_feature_dim × policy.n_obs_steps)
        action_dim      : normalized action dimension
        hidden_dims     : hidden layer widths for each ensemble member
        num_ensemble    : total ensemble members (default 7, from Uni-O4)
        num_elites      : members used for inference (default 5)
        device          : torch device string
    """

    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (200, 200, 200, 200),
        num_ensemble: int = 7,
        num_elites: int = 5,
        device: str = 'cuda',
    ) -> None:
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.model = EnsembleDynamicsModel(
            obs_dim=obs_feature_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_ensemble=num_ensemble,
            num_elites=num_elites,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_feature_dataset(
        self,
        policy,
        dataset,
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode the offline dataset once and build supervised training pairs:
            inputs  : [obs_feat_t | norm_action_t]       [N, obs_dim + act_dim]
            targets : [Δobs_feat_t→t+1 | reward_t]      [N, obs_dim + 1]

        next_obs_feat is obtained by encoding the observation at step t+1
        (shifting the obs sequence by 1 along the horizon axis).
        """
        from diffusion_policy_3d.common.pytorch_util import dict_apply
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_inputs, all_targets = [], []
        policy.eval()

        with torch.no_grad():
            for batch in loader:
                # Move tensors to device
                obs_dict = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['obs'].items()
                }
                action = policy.extract_action_chunk(batch['action'].to(self.device))
                reward = batch.get('reward', None)
                if reward is not None:
                    reward = reward.to(self.device)

                # Normalize
                nobs = policy.normalizer.normalize(obs_dict)
                if not policy.use_pc_color:
                    nobs['point_cloud'] = nobs['point_cloud'][..., :3]
                naction = policy.normalizer['action'].normalize(action).reshape(action.shape[0], -1)

                B = action.shape[0]

                # Current obs features  (steps 0 .. n_obs_steps-1)
                cur_nobs = dict_apply(
                    nobs,
                    lambda x: x[:, :policy.n_obs_steps].reshape(-1, *x.shape[2:])
                )
                obs_feat = policy.obs_encoder(cur_nobs).reshape(B, -1)  # [B, F]

                # Next obs features from the chunk-level next state supplied by the dataset.
                next_obs_raw = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['next_obs'].items()
                }
                nxt_nobs = policy.normalizer.normalize(next_obs_raw)
                if not policy.use_pc_color:
                    nxt_nobs['point_cloud'] = nxt_nobs['point_cloud'][..., :3]
                nxt_nobs = dict_apply(
                    nxt_nobs,
                    lambda x: x[:, :policy.n_obs_steps].reshape(-1, *x.shape[2:])
                )
                next_obs_feat = policy.obs_encoder(nxt_nobs).reshape(B, -1)  # [B, F]

                delta_feat = (next_obs_feat - obs_feat).cpu().numpy()
                inp = np.concatenate(
                    [obs_feat.cpu().numpy(), naction.cpu().numpy()], axis=-1)

                if reward is not None:
                    r = reward.cpu().numpy()
                    if r.ndim == 1:
                        r = r[:, None]
                else:
                    r = np.zeros((B, 1), dtype=np.float32)

                tgt = np.concatenate([delta_feat, r], axis=-1)
                all_inputs.append(inp.astype(np.float32))
                all_targets.append(tgt.astype(np.float32))

        return (np.concatenate(all_inputs, axis=0),
                np.concatenate(all_targets, axis=0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_dataset(
        self,
        policy,
        dataset,
        batch_size: int = 256,
        num_workers: int = 4,
        max_epochs: int = 200,
        max_epochs_since_update: int = 5,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
    ) -> Dict[str, object]:
        """
        Train the ensemble transition model.

        Steps:
          1. Build (input, target) dataset by encoding obs with frozen policy encoder.
          2. Fit StandardScaler on training inputs.
          3. NLL + logvar regularisation + weight decay training loop.
          4. Elite selection via holdout MSE; early stop when no improvement.

        Returns:
            Dictionary with final holdout loss and per-epoch train/val histories.
        """
        cprint("\n[TransitionModel] Encoding dataset for transition model training...", "cyan")
        inputs, targets = self._build_feature_dataset(
            policy, dataset, batch_size, num_workers)
        cprint(f"[TransitionModel] Dataset: {inputs.shape[0]} samples, "
               f"input_dim={inputs.shape[1]}, target_dim={targets.shape[1]}", "cyan")

        N = inputs.shape[0]
        holdout_size = min(int(N * holdout_ratio), 1000)
        train_size = N - holdout_size
        perm = np.random.permutation(N)
        train_in, train_tgt = inputs[perm[:train_size]], targets[perm[:train_size]]
        hold_in, hold_tgt = inputs[perm[train_size:]], targets[perm[train_size:]]

        self.scaler.fit(train_in)
        train_in_s = self.scaler.transform(train_in)
        hold_in_s = self.scaler.transform(hold_in)

        # Per-ensemble shuffled index arrays (shape [E, train_size])
        data_idxes = np.stack(
            [np.random.permutation(train_size)
             for _ in range(self.model.num_ensemble)], axis=0)

        def shuffle_rows(arr: np.ndarray) -> np.ndarray:
            order = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], order]

        holdout_losses = [1e10] * self.model.num_ensemble
        cnt = 0
        train_loss_history: List[float] = []
        val_loss_history: List[float] = []

        for epoch in range(max_epochs):
            train_loss = self._learn_epoch(
                train_in_s[data_idxes], train_tgt[data_idxes],
                batch_size, logvar_loss_coef)
            new_holdout_losses = self._validate(hold_in_s, hold_tgt)
            val_loss = float(np.sort(new_holdout_losses)[:self.model.num_elites].mean())
            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))

            data_idxes = shuffle_rows(data_idxes)

            improved = [
                i for i, (nl, ol) in enumerate(zip(new_holdout_losses, holdout_losses))
                if ol > 0 and (ol - nl) / ol > 0.01
            ]
            if improved:
                self.model.update_save(improved)
                holdout_losses = [
                    new_holdout_losses[i] if i in improved else holdout_losses[i]
                    for i in range(self.model.num_ensemble)
                ]
                cnt = 0
            else:
                cnt += 1

            if epoch % 20 == 0 or cnt >= max_epochs_since_update:
                cprint(f"[TransitionModel] Epoch {epoch:4d} | "
                       f"train={train_loss:.5f} | val={val_loss:.5f} | "
                       f"no-improve={cnt}/{max_epochs_since_update}", "green")

            if cnt >= max_epochs_since_update:
                break

        elites = sorted(range(len(holdout_losses)),
                        key=lambda i: holdout_losses[i])[:self.model.num_elites]
        self.model.set_elites(elites)
        self.model.load_save()
        self.model.eval()

        final_val = float(np.sort(holdout_losses)[:self.model.num_elites].mean())
        cprint(f"[TransitionModel] Training complete. "
               f"Elites={elites}, val_loss={final_val:.5f}", "green")
        return {
            'final_val_loss': final_val,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'elites': elites,
        }

    def _learn_epoch(
        self,
        inputs: np.ndarray,   # [E, train_size, in_dim]
        targets: np.ndarray,  # [E, train_size, out_dim]  (NOT scaled – raw delta)
        batch_size: int,
        logvar_loss_coef: float,
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for start in range(0, train_size, batch_size):
            x = torch.as_tensor(
                inputs[:, start:start + batch_size],
                dtype=torch.float32, device=self.device)   # [E, B, in_dim]
            y = torch.as_tensor(
                targets[:, start:start + batch_size],
                dtype=torch.float32, device=self.device)   # [E, B, out_dim]

            mean, logvar = self.model(x)   # [E, B, out_dim]
            inv_var = torch.exp(-logvar)
            mse_loss = (torch.pow(mean - y, 2) * inv_var).mean(dim=(1, 2)).sum()
            var_loss = logvar.mean(dim=(1, 2)).sum()
            loss = (mse_loss + var_loss
                    + self.model.get_decay_loss()
                    + logvar_loss_coef * self.model.max_logvar.sum()
                    - logvar_loss_coef * self.model.min_logvar.sum())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def _validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        mean, _ = self.model(x)
        loss = ((mean - y) ** 2).mean(dim=(1, 2))  # [E]
        return loss.cpu().numpy().tolist()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_next_features(
        self,
        obs_features: torch.Tensor,  # [B, obs_feature_dim]
        naction: torch.Tensor,        # [B, action_dim]
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next obs features and reward via one ensemble step.

        Mirrors EnsembleDynamics.step() from Uni-O4 but stays in torch.

        Returns:
            next_obs_features : [B, obs_feature_dim]
            pred_reward        : [B, 1]
        """
        self.model.eval()
        B = obs_features.shape[0]

        x = torch.cat([obs_features, naction], dim=-1)  # [B, in_dim]
        x = self.scaler.transform_tensor(x)             # normalise inputs

        mean, logvar = self.model(x)                     # [E, B, out_dim]
        elite_indices = self.model.elites.long()
        elite_mean = mean[elite_indices]

        if deterministic:
            selected = elite_mean.mean(dim=0)            # [B, out_dim]
        else:
            std = torch.exp(0.5 * logvar)
            samples = mean + torch.randn_like(std) * std     # [E, B, out_dim]

            # Select one elite per sample
            elite_idxs = torch.tensor(
                self.model.random_elite_idxs(B), device=self.device)  # [B]
            selected = samples[elite_idxs, torch.arange(B, device=self.device)]  # [B, out_dim]

        delta_feat = selected[:, :-1]           # [B, obs_feature_dim]
        pred_reward = selected[:, -1:]          # [B, 1]
        next_obs_features = obs_features + delta_feat

        return next_obs_features, pred_reward

    # ------------------------------------------------------------------
    # Checkpoint helpers (mirrors nn.Module interface for rl100_trainer)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler_mu': self.scaler.mu,
            'scaler_std': self.scaler.std,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.mu = state['scaler_mu']
        self.scaler.std = state['scaler_std']
        self.model.eval()
        cprint("[TransitionModel] Checkpoint loaded.", "green")
