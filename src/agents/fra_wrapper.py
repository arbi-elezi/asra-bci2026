"""FRA Wrapper — Orchestrates all FRA components around the frozen base policy.

This is the main integration point. It:
  1. Takes the frozen W_0 base policy
  2. Creates a perturbation copy W_t
  3. Runs the fear pipeline: cost critic → F_t^CA → TD-Fear → Uncertainty → FMS
  4. Applies DR (policy shaping) and FHR (homeostatic regulation) to W_t
  5. Runs SCL (action mixing) on the output
  6. Feeds GTCC labels to FC
  7. Logs all per-timestep metrics (M3, M13 mandatory for C2)

CRITICAL INVARIANT: W_0 is NEVER modified. Only W_t is perturbed.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from src.agents.ppo_base import ActorCritic
from src.agents.cost_critic import CostCriticNet, compute_fear_signal_ca
from src.components.fhr import FHR, FHRConfig
from src.components.dr import DR, DRConfig
from src.components.td_fear import TDFear, TDFearConfig
from src.components.scl import SCL, SCLConfig
from src.components.gtcc import GTCC, GTCCConfig
from src.components.fc import FC, FCConfig
from src.components.fms import FMS, FMSConfig


@dataclass
class FRAConfig:
    """Master FRA configuration."""
    enabled: bool = True
    fhr: FHRConfig = field(default_factory=FHRConfig)
    dr: DRConfig = field(default_factory=DRConfig)
    td_fear: TDFearConfig = field(default_factory=TDFearConfig)
    scl: SCLConfig = field(default_factory=SCLConfig)
    gtcc: GTCCConfig = field(default_factory=GTCCConfig)
    fc: FCConfig = field(default_factory=FCConfig)
    fms: FMSConfig = field(default_factory=FMSConfig)
    # Uncertainty
    uncertainty_enabled: bool = True
    uncertainty_beta: float = 1.0
    mc_dropout_passes: int = 5
    # Fear signal
    lambda_a: float = 5.0  # Sigmoid scaling for F_t^CA


class FRAWrapper:
    """Fear-Regulated Agent — wraps a frozen PPO base policy.

    Usage:
        w0_model = load_model(...)  # Frozen base policy
        fra = FRAWrapper(w0_model, cost_critic, config)
        for episode:
            fra.reset()
            obs = env.reset()
            for step:
                action, info = fra.step(obs, env_cost, env_ttc)
                obs, reward, done, _, env_info = env.step(action)
    """

    def __init__(
        self,
        w0_model: ActorCritic,
        cost_critic: CostCriticNet,
        config: FRAConfig,
        device: str = "cuda",
    ) -> None:
        self.cfg = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ── W_0: FROZEN base policy (NEVER modified) ──
        self.w0_model = copy.deepcopy(w0_model).to(self.device)
        self.w0_model.eval()
        for p in self.w0_model.parameters():
            p.requires_grad = False
        self._w0_hash = self._compute_hash(self.w0_model)

        # W_0 parameter snapshot for FHR
        self.w0_params: dict[str, torch.Tensor] = {
            name: p.data.clone().detach()
            for name, p in self.w0_model.actor.named_parameters()
        }

        # ── W_t: Perturbation copy (actively modified by DR/FHR) ──
        self.wt_model = copy.deepcopy(w0_model).to(self.device)
        self.wt_model.train()  # Needs gradients for DR

        # ── Cost critic (frozen) ──
        self.cost_critic = cost_critic.to(self.device)

        # ── Components ──
        self.fhr = FHR(config.fhr, self.w0_params, device=str(self.device))
        self.dr = DR(config.dr, device=str(self.device))
        self.td_fear = TDFear(config.td_fear)
        self.scl = SCL(config.scl)
        self.gtcc = GTCC(config.gtcc)
        self.fc = FC(config.fc, device=str(self.device))
        self.fms = FMS(config.fms, device=str(self.device))

        # ── Per-timestep metrics (M3, M13 mandatory for C2) ──
        self._step_metrics: list[dict[str, float]] = []
        self._episode_count: int = 0

    def step(
        self,
        obs: np.ndarray | torch.Tensor,
        cost: float,
        ttc: float,
        obstacle_class: int = -1,
    ) -> tuple[int, dict[str, Any]]:
        """Execute one FRA step.

        Args:
            obs: Current observation (R^12).
            cost: Observable cost signal c_t ∈ [0, 1].
            ttc: Time-to-collision.
            obstacle_class: Nearest obstacle class (for M10-s).

        Returns:
            (action, step_info_dict)
        """
        if not self.cfg.enabled:
            # No FRA — just use W_0
            return self._base_action(obs), {"fear": 0.0, "alpha": 0.0}

        # Convert obs to tensor
        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device)

        # ── 1. Fear pipeline ──

        # 1a. Get greedy action from current W_t
        with torch.no_grad():
            dist_wt, _ = self.wt_model.forward(obs_t.unsqueeze(0))
            greedy_action = dist_wt.probs.argmax(dim=-1).item()

        # 1b. Cost-advantage fear F_t^CA (Equation 11)
        f_ca = compute_fear_signal_ca(
            self.cost_critic, obs_t, greedy_action,
            lambda_a=self.cfg.lambda_a,
        )

        # 1c. TD-Fear smoothing
        f_td = self.td_fear.step(f_ca)

        # 1d. Epistemic uncertainty (MC dropout)
        if self.cfg.uncertainty_enabled:
            u_t = self._compute_uncertainty(obs_t)
            f_uncertain = f_td * (1 + self.cfg.uncertainty_beta * u_t)
            f_uncertain = max(0.0, min(1.0, f_uncertain))
        else:
            f_uncertain = f_td

        # 1e. FMS distribution shift correction
        predicted_cost = self.cost_critic.compute_advantage(obs_t, greedy_action) + 0.5
        delta_fms = self.fms.step(obs_t, predicted_cost)
        f_final = max(0.0, min(1.0, f_uncertain + delta_fms))

        # ── 2. FC classification + GTCC ──
        fc_pred = self.fc.predict(obs_t)
        gtcc_result = self.gtcc.step(obs_t, cost, ttc)
        if gtcc_result["label"] is not None:
            self.fc.add_label(obs_t, gtcc_result["label"])

        # ── 3. Safe action (Definition 2) ──
        safe_action = self.dr.compute_safe_action(
            self.cost_critic, obs_t, n_actions=4
        )

        # ── 4. DR policy shaping ──
        dr_updates = self.dr.step(self.wt_model, obs_t, f_final, safe_action)

        # ── 5. FHR restoring force ──
        actor_params = list(self.wt_model.actor.named_parameters())
        fhr_updates = self.fhr.step(self.wt_model, actor_params)

        # 5b. BC term (if enabled)
        bc_updates = self.fhr.compute_bc_update(
            self.wt_model, self.w0_model, obs_t.unsqueeze(0)
        )

        # ── 6. Apply updates to W_t ──
        with torch.no_grad():
            for name, param in self.wt_model.actor.named_parameters():
                total_update = torch.zeros_like(param.data)

                if name in dr_updates:
                    total_update += dr_updates[name]
                if name in fhr_updates:
                    total_update += fhr_updates[name]
                if name in bc_updates:
                    total_update += bc_updates[name]

                param.data += total_update

        # ── 7. SCL action mixing ──
        with torch.no_grad():
            dist_wt_new, _ = self.wt_model.forward(obs_t.unsqueeze(0))
            policy_probs = dist_wt_new.probs.squeeze(0)

        action, alpha = self.scl.step(policy_probs, safe_action, f_final)

        # ── 8. Compute metrics ──
        wdn = self.fhr.compute_wdn(actor_params)

        # KL divergence M9
        with torch.no_grad():
            dist_w0, _ = self.w0_model.forward(obs_t.unsqueeze(0))
            kl = torch.distributions.kl_divergence(
                dist_w0, dist_wt_new
            ).item() if hasattr(dist_w0, 'probs') else 0.0

        step_info = {
            "fear_ca": f_ca,
            "fear_td": f_td,
            "fear_final": f_final,
            "wdn": wdn,
            "kl_divergence": kl,
            "alpha": alpha,
            "safe_action": safe_action,
            "fc_prediction": fc_pred,
            "gtcc_label": gtcc_result["label"],
            "fms_delta": delta_fms,
            "greedy_action": greedy_action,
            "selected_action": action,
        }

        # Get DR grad norm for M13
        dr_metrics = self.dr.get_metrics()
        if dr_metrics["grad_norms"]:
            step_info["grad_norm"] = dr_metrics["grad_norms"][-1]
            step_info["a1_violation"] = dr_metrics["grad_norms"][-1] > self.dr.cfg.g_max

        self._step_metrics.append(step_info)

        # ── 9. Verify W_0 immutability ──
        assert self._compute_hash(self.w0_model) == self._w0_hash, \
            "CRITICAL BUG: W_0 has been modified!"

        return action, step_info

    def _base_action(self, obs: np.ndarray | torch.Tensor) -> int:
        """Get action from frozen W_0 (for C1 baseline)."""
        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device)

        with torch.no_grad():
            dist, _ = self.w0_model.forward(obs_t.unsqueeze(0))
            return dist.sample().item()

    def _compute_uncertainty(self, obs: torch.Tensor) -> float:
        """Compute epistemic uncertainty via MC dropout.

        Returns variance of action probabilities across MC passes.
        """
        self.wt_model.train()  # Enable dropout
        probs_list = []
        for _ in range(self.cfg.mc_dropout_passes):
            with torch.no_grad():
                dist, _ = self.wt_model.forward(obs.unsqueeze(0))
                probs_list.append(dist.probs.squeeze(0))

        probs_stack = torch.stack(probs_list)  # [mc_passes, n_actions]
        variance = probs_stack.var(dim=0).mean().item()
        return variance

    def _compute_hash(self, model: nn.Module) -> str:
        """SHA-256 hash of model parameters."""
        hasher = hashlib.sha256()
        for p in model.parameters():
            hasher.update(p.data.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def get_episode_metrics(self) -> dict[str, Any]:
        """Get all per-timestep metrics for the current episode."""
        return {
            "step_metrics": self._step_metrics.copy(),
            "fhr": self.fhr.get_metrics(),
            "dr": self.dr.get_metrics(),
            "td_fear": self.td_fear.get_metrics(),
            "scl": self.scl.get_metrics(),
            "gtcc": self.gtcc.get_metrics(),
            "fc": self.fc.get_metrics(),
            "fms": self.fms.get_metrics(),
            "w0_hash": self._w0_hash,
        }

    def reset(self) -> None:
        """Reset all components for a new episode."""
        # Reset W_t back to W_0 at episode start
        self.wt_model.load_state_dict(self.w0_model.state_dict())
        self.wt_model.train()

        # Reset components
        self.fhr.reset()
        self.dr.reset()
        self.td_fear.reset()
        self.scl.reset()
        self.gtcc.reset()
        self.fc.reset()
        self.fms.reset()

        self._step_metrics = []
        self._episode_count += 1

    def get_w0_hash(self) -> str:
        """Return W_0 hash for reproducibility verification."""
        return self._w0_hash
