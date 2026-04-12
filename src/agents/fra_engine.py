"""FRA Perturbation Engine — Orchestrates fear-driven LLM weight perturbation.

This is the master integration layer. It:
  1. Receives fear signal F_t from the INDEPENDENT fear detector
  2. When F_t spikes: DR perturbs LLM's LoRA weights toward safe action
  3. Continuously: FHR pulls LoRA weights back toward W_0 (homeostasis)
  4. The perturbation propagates through the LLM in WAVES:
     - Immediate: action head weights shift first
     - Delayed: LoRA adapter weights shift next
     - Recovery: FHR slowly restores all to W_0

Scientific method:
  - Every step logs M3 (WDN) and M13 (gradient norm) for C2
  - W_0 hash verified at every episode boundary
  - All randomness via seeded generators
  - Bootstrap CIs computed over matched seeds

Proposition 1 applies because:
  - LoRA params are the perturbable set (replaces PPO actor params)
  - Fisher diagonal computed over LoRA params
  - ||W_t - W_0|| = ||LoRA_t - LoRA_0|| bounded by same recurrence
  - FHR restoring force: η_h · F̂_I ⊙ (W_0 - W_t) on LoRA params
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from src.agents.llm_driver import LLMDriver
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.components.td_fear import TDFear, TDFearConfig
from src.components.scl import SCL, SCLConfig
from src.components.gtcc import GTCC, GTCCConfig
from src.components.fc import FC, FCConfig
from src.components.fms import FMS, FMSConfig


@dataclass
class FRAEngineConfig:
    """Master configuration for the FRA perturbation engine."""
    enabled: bool = True

    # DR parameters — FAST perturbation when fear spikes
    eta_f: float = 0.01        # Fear learning rate (FAST — ~1000x η_h)
    dr_epsilon: float = 0.05   # DR activation threshold
    g_max: float = 10.0        # A1 gradient norm bound

    # FHR parameters — SLOW, GRADUAL homeostatic rebound
    # Homeostasis is NOT instant. It is a slow drift back to W_0 over many steps.
    # The asymmetry η_f >> η_h creates: fast spike → slow recovery
    # Recovery timescale: ~1/(η_h · f_min) steps
    # With η_h=1e-5, f_min=0.01: ~10M steps (very slow)
    # Visible recovery over 50-200 episode steps requires careful tuning.
    eta_h: float = 1e-5        # Homeostatic rate (SLOW — ~1000x smaller than η_f)
    fhr_mode: str = "fisher"   # "fisher" or "l2" (C3a ablation)
    bc_enabled: bool = True    # BC term (C3b ablation)
    eta_bc: float = 1e-5       # BC rate (ALSO slow — part of homeostasis)

    # Component toggles (for ablations)
    dr_enabled: bool = True      # C6: disable
    fhr_enabled: bool = True
    fc_enabled: bool = True
    fc_frozen: bool = False      # C4: freeze
    td_fear_enabled: bool = True
    fms_enabled: bool = True     # C5: disable
    gtcc_enabled: bool = True
    scl_enabled: bool = True
    fear_detector_enabled: bool = True

    # Fear pipeline
    td_fear: TDFearConfig = field(default_factory=TDFearConfig)
    scl: SCLConfig = field(default_factory=SCLConfig)
    gtcc: GTCCConfig = field(default_factory=GTCCConfig)
    fc: FCConfig = field(default_factory=FCConfig)
    fms: FMSConfig = field(default_factory=FMSConfig)
    fear_detector: FearDetectorConfig = field(default_factory=FearDetectorConfig)

    # Wave perturbation parameters
    wave_decay: float = 0.9   # Fear wave propagation decay per layer

    # Temporal gradient homeostasis parameters
    #
    # Scientific basis (3 independent justifications):
    #
    # 1. NEUROSCIENCE: HPA axis cortisol recovery follows a delayed, gradual
    #    profile — fast spike, slow return. (Sapolsky 2004; de Kloet 2005;
    #    LeDoux 2000 [9]; Damasio 1994 [10]; Lindroos 2024 [11])
    #
    # 2. CONTROL THEORY: Equivalent to a PI (proportional-integral) controller
    #    where the integral term accumulates restoring pressure over time.
    #    Standard adaptive gain scheduling in control systems.
    #
    # 3. FORGETTING LITERATURE: Kirkpatrick [30] (EWC) shows timing of
    #    regularization matters — immediate strong regularization prevents
    #    adaptation; delayed increasing regularization allows temporary
    #    adaptation then recovery.
    #
    # Recovery: η_h_eff(t) = η_h_base · min(1 + γ · t_since_spike, max_mult)
    # - Immediately after spike: weak (allows perturbation to take effect)
    # - Over time: grows stronger (prevents indefinite drift)
    # - Preserves Proposition 1: uses minimum η_h for bound, actual is tighter
    #
    # Falsifiable via H3a (single-spike recovery) and H3b (accumulation).
    #
    gamma_reascent: float = 0.02   # Recovery doubles after ~50 steps (1/γ)
    max_reascent_multiplier: float = 10.0  # Cap at 10× base rate (~450 steps)


class FRAEngine:
    """FRA Perturbation Engine — fear spikes → weight waves → homeostatic rebound.

    The core loop each timestep:
      1. Fear detector produces F_t (independent of LLM)
      2. TD-Fear smooths F_t temporally
      3. FMS corrects for distribution shift
      4. If F_t > ε: DR computes gradient toward safe action
      5. DR gradient propagates through LoRA params in WAVES
      6. FHR continuously pulls all params toward W_0
      7. SCL mixes action-level output for immediate safety
      8. GTCC generates training labels for FC
    """

    def __init__(
        self,
        llm_driver: LLMDriver,
        config: FRAEngineConfig,
    ) -> None:
        self.cfg = config
        self.llm = llm_driver
        self.device = llm_driver.device

        # Get W_0 snapshot
        self.w0 = llm_driver.get_w0_params()

        # Fisher diagonal (computed offline, set later)
        self.fisher_diag: torch.Tensor | None = None

        # ── Components ──
        self.fear_detector = FearDetector(config.fear_detector)
        self.td_fear = TDFear(config.td_fear)
        self.scl = SCL(config.scl)
        self.gtcc = GTCC(config.gtcc)
        self.fc = FC(config.fc, device=str(self.device))
        self.fms = FMS(config.fms, device=str(self.device))

        # ── Per-timestep metrics (M3, M13 — mandatory for C2) ──
        self._step_metrics: list[dict[str, float]] = []
        self._episode_count: int = 0

        # ── DR gradient tracking ──
        self._grad_norms: list[float] = []
        self._a1_violations: int = 0

        # ── Temporal gradient homeostasis state ──
        # Tracks steps since last significant fear spike per parameter group
        self._steps_since_spike: int = 0
        self._last_fear_was_spike: bool = False

    def step(
        self,
        obs: np.ndarray,
        cost: float,
        ttc: float,
        obstacle_class: int,
        state_text_fn,
    ) -> tuple[int, dict[str, Any]]:
        """Execute one FRA-augmented decision step.

        Args:
            obs: R^12 observation from highway-env.
            cost: Observable cost signal c_t ∈ [0, 1].
            ttc: Time-to-collision.
            obstacle_class: Nearest obstacle class (0, 1, 2).
            state_text_fn: Function obs → text for LLM.

        Returns:
            (action, step_info)
        """
        if not self.cfg.enabled:
            # No FRA — raw LLM action
            action, probs = self.llm.get_action(state_text_fn(obs))
            return action, {"fear": 0.0, "wdn": 0.0}

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # ── 1. INDEPENDENT FEAR DETECTION ──
        # The fear model is SEPARATE from the LLM
        with torch.no_grad():
            # Get LLM's current greedy action (for cost-advantage)
            logits = self.llm.get_action_logits(state_text_fn(obs))
            greedy_action = logits.argmax().item()

        fear_raw, fear_components = self.fear_detector.detect(
            obs, cost, ttc, greedy_action
        )

        # ── 2. TD-FEAR SMOOTHING ──
        fear_smoothed = self.td_fear.step(fear_raw)

        # ── 3. FMS DISTRIBUTION SHIFT CORRECTION ──
        if self.cfg.fms_enabled:
            fms_delta = self.fms.step(obs_t, cost)
            fear_corrected = max(0.0, min(1.0, fear_smoothed + fms_delta))
        else:
            fms_delta = 0.0
            fear_corrected = fear_smoothed

        fear_final = fear_corrected

        # ── 4. COMPUTE SAFE ACTION ──
        safe_action = self._compute_safe_action(obs_t)

        # ── 5. DR: FEAR-TRIGGERED WEIGHT PERTURBATION (WAVE) ──
        grad_norm = 0.0
        if self.cfg.dr_enabled and fear_final > self.cfg.dr_epsilon:
            grad_norm = self._apply_dr_wave(state_text_fn(obs), safe_action, fear_final)
            self._steps_since_spike = 0  # Reset recovery clock
            self._last_fear_was_spike = True
        else:
            self._steps_since_spike += 1
            self._last_fear_was_spike = False

        # ── 6. FHR: GRADUAL HOMEOSTATIC REBOUND ──
        # Recovery follows temporal gradient re-ascent:
        # Weak immediately after spike, GROWS stronger over time
        if self.cfg.fhr_enabled:
            self._apply_fhr_gradual()

        # ── 7. BC TERM ──
        if self.cfg.bc_enabled:
            self._apply_bc(state_text_fn(obs))

        # ── 8. GET POST-PERTURBATION ACTION ──
        new_logits = self.llm.get_action_logits(state_text_fn(obs))
        new_probs = torch.softmax(new_logits, dim=-1)

        # ── 9. SCL: ACTION-LEVEL MIXING ──
        if self.cfg.scl_enabled:
            action, alpha = self.scl.step(new_probs, safe_action, fear_final)
        else:
            action = new_probs.argmax().item()
            alpha = 0.0

        # ── 10. GTCC + RLAIF + FC ──
        if self.cfg.gtcc_enabled:
            # Pass full context to GTCC for RLAIF judge
            closing_speed = max(0.0, obs[2] - obs[8]) if len(obs) > 8 else 0.0  # ego_vx - obs_vx
            gap = abs(obs[6]) if len(obs) > 6 else 100.0  # nearest_dx
            gtcc_result = self.gtcc.step(
                obs_t, cost, ttc,
                obstacle_class=obstacle_class,
                ego_speed=obs[2] if len(obs) > 2 else 25.0,
                closing_speed=closing_speed,
                gap=gap,
            )
            if gtcc_result["label"] is not None and self.cfg.fc_enabled:
                self.fc.add_label(obs_t, gtcc_result["label"])

        # ── 11. METRICS ──
        wdn = self._compute_wdn()

        # Track A1 violations (M13)
        self._grad_norms.append(grad_norm)
        if grad_norm > self.cfg.g_max:
            self._a1_violations += 1

        step_info = {
            "fear_raw": fear_raw,
            "fear_smoothed": fear_smoothed,
            "fear_final": fear_final,
            "fear_components": fear_components,
            "wdn": wdn,  # M3
            "grad_norm": grad_norm,  # M13
            "a1_violation": grad_norm > self.cfg.g_max,
            "alpha": alpha,
            "safe_action": safe_action,
            "selected_action": action,
            "fms_delta": fms_delta,
            "cost": cost,
            "ttc": ttc,
        }
        self._step_metrics.append(step_info)

        return action, step_info

    def _apply_dr_wave(
        self, state_text: str, safe_action: int, fear: float
    ) -> float:
        """Apply DR gradient in WAVES through the LoRA parameters.

        The wave metaphor:
        - Action head gets the STRONGEST perturbation (immediate layer)
        - Each LoRA layer deeper gets a DECAYED perturbation
        - This creates a wave-like propagation of fear through the network

        Returns: gradient norm (for M13).
        """
        # Compute gradient: ∇_W log π(a_safe | s)
        self.llm.model.zero_grad()
        self.llm.action_head.zero_grad()

        logits = self.llm.get_action_logits(state_text)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob_safe = log_probs[safe_action]
        log_prob_safe.backward()

        # Apply DR update with wave decay
        grad_norm_sq = 0.0
        wave_scale = 1.0
        perturbable = self.llm.get_perturbable_params()

        with torch.no_grad():
            for name, param in perturbable:
                if param.grad is None:
                    continue

                g = param.grad.data
                grad_norm_sq += (g.float() ** 2).sum().item()

                # DR update: η_f · F_t · g · wave_scale
                # Increases log-probability of safe action
                update = self.cfg.eta_f * fear * g * wave_scale
                param.data += update.to(param.dtype)

                # Wave decay for deeper layers
                if "lora" in name:
                    wave_scale *= self.cfg.wave_decay

        return float(np.sqrt(grad_norm_sq))

    def _apply_fhr_gradual(self) -> None:
        """Apply Fisher-weighted homeostatic restoring force with TEMPORAL GRADIENT.

        The restoring force follows a temporal gradient re-ascent:
          η_h_effective(t) = η_h_base · min(1 + γ · t_since_spike, max_mult)

        Behavior:
        - Immediately after fear spike: η_h_eff ≈ η_h_base (weak — almost no recovery)
        - 10 steps later: η_h_eff ≈ η_h_base · 1.2 (slightly stronger)
        - 50 steps later: η_h_eff ≈ η_h_base · 2.0 (moderate recovery)
        - 200 steps later: η_h_eff ≈ η_h_base · 5.0 (strong pull back)

        This creates the biologically-inspired asymmetry:
          FAST spike → GRADUAL, ACCELERATING recovery

        ΔW = η_h_eff(t) · F̂_I ⊙ (W_0 - W_t)
        """
        # Temporal gradient re-ascent: recovery force grows over time
        temporal_scale = min(
            1.0 + self.cfg.gamma_reascent * self._steps_since_spike,
            self.cfg.max_reascent_multiplier,
        )
        eta_h_effective = self.cfg.eta_h * temporal_scale

        perturbable = self.llm.get_perturbable_params()
        fisher_idx = 0

        with torch.no_grad():
            for name, param in perturbable:
                w0_val = self.w0.get(name)
                if w0_val is None:
                    continue

                diff = w0_val.to(param.dtype) - param.data

                if self.cfg.fhr_mode == "fisher" and self.fisher_diag is not None:
                    # Fisher-weighted restoring
                    n_elem = param.numel()
                    f_slice = self.fisher_diag[fisher_idx:fisher_idx + n_elem]
                    fisher_idx += n_elem
                    f_reshaped = f_slice.reshape(param.shape).to(param.dtype)
                    update = eta_h_effective * f_reshaped * diff
                else:
                    # L2 restoring (for C3a ablation)
                    update = eta_h_effective * diff

                param.data += update

    def _apply_bc(self, state_text: str) -> None:
        """Apply behavioral cloning term.

        ΔW = η_bc · ∇_W D_KL(π_{W_0} || π_{W_t})

        Penalizes behavioral divergence from the base policy.
        Implementation: save W_t → restore W_0 → compute target → restore W_t → gradient step.
        """
        if not self.cfg.bc_enabled:
            return

        # 1. Save current W_t
        wt_snapshot = {}
        for name, param in self.llm.get_perturbable_params():
            wt_snapshot[name] = param.data.clone()

        # 2. Compute π_{W_0} distribution (target for BC)
        self.llm.restore_to_w0()
        with torch.no_grad():
            w0_logits = self.llm.get_action_logits(state_text)
            w0_probs = torch.softmax(w0_logits, dim=-1).detach()

        # 3. Restore W_t
        for name, param in self.llm.get_perturbable_params():
            if name in wt_snapshot:
                param.data.copy_(wt_snapshot[name])

        # 4. Compute ∇_W D_KL(π_{W_0} || π_{W_t}) w.r.t. W_t
        self.llm.model.zero_grad()
        self.llm.action_head.zero_grad()

        wt_logits = self.llm.get_action_logits(state_text)
        wt_log_probs = torch.log_softmax(wt_logits, dim=-1)
        # D_KL(p || q) = Σ p * (log p - log q)
        kl_div = torch.sum(w0_probs * (torch.log(w0_probs + 1e-8) - wt_log_probs))
        kl_div.backward()

        # 5. Apply BC update: step toward reducing D_KL
        with torch.no_grad():
            for name, param in self.llm.get_perturbable_params():
                if param.grad is not None:
                    param.data -= self.cfg.eta_bc * param.grad.data

    def _compute_safe_action(self, obs: torch.Tensor) -> int:
        """Compute safe action from environment state.

        Simple heuristic aligned with Definition 2:
        - If TTC < 2: BRAKE (action 2)
        - If obstacle in lane: LANE_CHANGE (action 3)
        - Otherwise: MAINTAIN (action 0)
        """
        ttc_feature = obs[10] if obs.shape[0] > 10 else 10.0
        if isinstance(ttc_feature, torch.Tensor):
            ttc_feature = ttc_feature.item()

        if ttc_feature < 2.0:
            return 2  # BRAKE
        elif ttc_feature < 4.0:
            return 3  # LANE_CHANGE
        else:
            return 0  # MAINTAIN

    def _compute_wdn(self) -> float:
        """Compute Weight Deviation Norm: ||W_t - W_0||_F.

        MUST be in float64 for accumulation (coding rule).
        """
        total = 0.0
        for name, param in self.llm.get_perturbable_params():
            w0_val = self.w0.get(name)
            if w0_val is None:
                continue
            diff = param.data.float().double() - w0_val.float().double()
            total += (diff * diff).sum().item()
        return float(np.sqrt(total))

    def compute_fisher_diagonal(self, d_ref_texts: list[str]) -> torch.Tensor:
        """Compute diagonal Fisher information over LoRA + action head params.

        F̂_I = (1/N) Σ (∇_W log π_W(a|s))^2

        Only computed over perturbable (LoRA + action head) parameters.

        Args:
            d_ref_texts: List of state text descriptions from D_ref.

        Returns:
            Fisher diagonal tensor.
        """
        perturbable = self.llm.get_perturbable_params()
        n_params = sum(p.numel() for _, p in perturbable)
        fisher = torch.zeros(n_params, dtype=torch.float64, device=self.device)

        for text in d_ref_texts:
            self.llm.model.zero_grad()
            self.llm.action_head.zero_grad()

            logits = self.llm.get_action_logits(text)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob.backward()

            grads = []
            for _, param in perturbable:
                if param.grad is not None:
                    grads.append(param.grad.data.float().flatten())
                else:
                    grads.append(torch.zeros(param.numel(), device=self.device))
            grad_vec = torch.cat(grads).double()
            fisher += grad_vec ** 2

        fisher /= len(d_ref_texts)

        # Ensure f_min > 0 (A2)
        fisher = torch.clamp(fisher, min=1e-8)

        self.fisher_diag = fisher
        return fisher

    def get_episode_metrics(self) -> dict[str, Any]:
        """Collect all metrics for current episode."""
        return {
            "step_metrics": self._step_metrics.copy(),
            "fear_detector": self.fear_detector.get_metrics(),
            "td_fear": self.td_fear.get_metrics(),
            "scl": self.scl.get_metrics(),
            "gtcc": self.gtcc.get_metrics(),
            "fc": self.fc.get_metrics(),
            "fms": self.fms.get_metrics(),
            "a1_violations": self._a1_violations,
            "a1_violation_fraction": (
                self._a1_violations / max(len(self._grad_norms), 1)
            ),
            "grad_norms": self._grad_norms.copy(),
            "w0_hash": self.llm.get_w0_hash(),
        }

    def reset(self) -> None:
        """Reset for new episode."""
        self.llm.restore_to_w0()
        self.fear_detector.reset()
        self.td_fear.reset()
        self.scl.reset()
        self.gtcc.reset()
        self.fc.reset()
        self.fms.reset()
        self._step_metrics = []
        self._grad_norms = []
        self._a1_violations = 0
        self._steps_since_spike = 0
        self._last_fear_was_spike = False
        self._episode_count += 1
