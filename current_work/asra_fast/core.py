"""ASRA-fast: architecture-agnostic risk-performance modulation on a frozen MLP policy."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Frozen policy: small MLP actor (12 -> h -> h -> 4). 5,252 params at h=64,
# ----------------------------------------------------------------------------- #
class MLPActor(nn.Module):
    def __init__(self, obs_dim: int = 12, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------------- #
# Hand-designed risk evaluator (identical rubric to the submitted run_multi_trial).
# Actions: 0=maintain 1=accelerate 2=brake 3=lane-change
# ----------------------------------------------------------------------------- #
def compute_risk(cost: float, ttc: float, action: int) -> float:
    base = {0: 0.3, 1: 0.6, 2: 0.05, 3: 0.5}.get(action, 0.3)
    if action != 2 and ttc < 3.0:
        base = min(1.0, base + (3.0 - ttc) / 3.0 * 0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2:
        base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


# ----------------------------------------------------------------------------- #
# Salience S_t. Two implementations:
#   - 'cost'    : S = cost (TTC-derived); fast, no training. Used for smoke tests.
#                 tunable weights (w_ae, w_if, w_ca) -> enables the Eq.(1) ablation.
# ----------------------------------------------------------------------------- #
@dataclass
class SalienceConfig:
    kind: str = "cost"            # 'cost' or 'ensemble'
    w_ae: float = 0.3
    w_if: float = 0.2
    w_ca: float = 0.5
    ca_lambda: float = 5.0


class CostSalience:
    """Trivial salience: S = TTC-derived cost in [0,1]."""
    def __init__(self, cfg: SalienceConfig):
        self.cfg = cfg

    def __call__(self, obs: np.ndarray, cost: float, ttc: float) -> float:
        return float(np.clip(cost, 0.0, 1.0))


# ----------------------------------------------------------------------------- #
# Operates per-parameter-tensor: epicenter = argmax|grad|, smooth decay outward.
# ----------------------------------------------------------------------------- #
def apply_gaussian_(param: torch.Tensor, grad: torch.Tensor,
                    epicenter: int, sigma_frac: float, magnitude: float) -> None:
    n = param.numel()
    idx = torch.arange(n, device=param.device, dtype=torch.float32)
    sig_abs = max(1.0, sigma_frac * n)
    kernel = torch.exp(-((idx - epicenter) ** 2) / (2 * sig_abs ** 2))
    param.data.view(-1).sub_(magnitude * kernel * grad.reshape(-1))


# ----------------------------------------------------------------------------- #
# ASRA controller (fixed-gain). One scalar `gain` scales the whole intervention.
# Channels are individually toggleable for the 2x2 factorial.
# ----------------------------------------------------------------------------- #
@dataclass
class ASRAConfig:
    gain: float = 1.0                 # the frontier knob g
    weight_channel: bool = True       # Gaussian weight perturbation
    conf_channel: bool = True         # per-action logit suppression
    select: str = "greedy"            # 'greedy' | 'sample' | 'bounded'
    alpha_thresh: float = 0.05        # activate when S*R exceeds this
    # weight channel
    eta_w: float = 0.05               # base perturbation magnitude scale
    sigma_frac: float = 0.05          # Gaussian width as fraction of tensor size
    # confidence channel
    kappa: float = 4.0                # logit-suppression scale (per unit g*alpha)
    temp_max: float = 2.5             # for 'bounded': T cap
    topp: float = 0.9                 # for 'bounded': nucleus filter
    # recovery
    rho_conf: float = 0.92            # suppression decay
    eta_h: float = 0.1                # Fisher-weighted weight recovery rate
    # targeting ablation
    targeting: str = "gradient"       # 'gradient' | 'random' | 'uniform'


class ASRA:
    """Fixed-gain ASRA wrapper around a frozen MLP actor.

    Usage per episode:
        asra.reset()                       # clears accumulated suppression
        ... restore actor to W_0 first ...
        for each step:
            action, diag = asra.act(actor, obs, cost, ttc)
    The actor's parameters are perturbed in place and pulled back toward W_0 each
    step via Fisher-weighted recovery. Caller restores W_0 between episodes.
    """
    def __init__(self, cfg: ASRAConfig, w0: dict[str, torch.Tensor],
                 fisher: dict[str, torch.Tensor], salience: Callable,
                 device: str = "cpu", rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.w0 = w0
        self.fisher = fisher
        self.salience = salience
        self.device = torch.device(device)
        self.rng = rng or np.random.default_rng(0)
        self.n_actions = 4
        self.sup = np.zeros(self.n_actions, dtype=np.float64)
        self.active_steps = 0
        self.lessrisky_hits = 0
        self.risk_red_sum = 0.0

    def reset(self) -> None:
        self.sup[:] = 0.0
        self.active_steps = 0
        self.lessrisky_hits = 0
        self.risk_red_sum = 0.0

    def _select(self, logits: torch.Tensor) -> int:
        cfg = self.cfg
        if cfg.select == "greedy":
            return int(torch.argmax(logits).item())
        if cfg.select == "sample":
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        # bounded: cap temperature + nucleus filter, then sample (entropy-controlled)
        probs = torch.softmax(logits / cfg.temp_max, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sp, dim=-1)
        keep = cdf <= cfg.topp
        keep[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[si[keep]] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    def act(self, actor: MLPActor, obs: np.ndarray, cost: float, ttc: float) -> tuple[int, dict]:
        cfg = self.cfg
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # current (possibly perturbed) logits, greedy action
        with torch.no_grad():
            base_logits = actor(obs_t).squeeze(0)
        greedy = int(torch.argmax(base_logits).item())

        S = float(self.salience(obs, cost, ttc))
        R = compute_risk(cost, ttc, greedy)
        alpha = S * R
        diag = {"S": S, "R": R, "alpha": alpha, "active": 0}

        if alpha > cfg.alpha_thresh and cfg.gain > 0:
            self.active_steps += 1
            diag["active"] = 1

            # ---- Channel 1: targeted Gaussian weight perturbation ----
            if cfg.weight_channel:
                actor.zero_grad(set_to_none=True)
                logits_g = actor(obs_t).squeeze(0)
                F.log_softmax(logits_g, dim=-1)[greedy].backward()
                with torch.no_grad():
                    for name, p in actor.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        n = p.numel()
                        if cfg.targeting == "gradient":
                            epi = int(g.abs().view(-1).argmax().item())
                            apply_gaussian_(p, g, epi, cfg.sigma_frac,
                                            cfg.eta_w * cfg.gain * alpha)
                        elif cfg.targeting == "random":
                            epi = int(self.rng.integers(0, n))
                            apply_gaussian_(p, g, epi, cfg.sigma_frac,
                                            cfg.eta_w * cfg.gain * alpha)
                        else:  # uniform: perturb all weights equally (no kernel)
                            p.data.view(-1).sub_(cfg.eta_w * cfg.gain * alpha * g.reshape(-1))
                actor.zero_grad(set_to_none=True)

            # ---- Channel 2: per-action confidence suppression ----
            if cfg.conf_channel:
                # suppress the (risky) greedy action proportional to g*alpha
                self.sup[greedy] -= cfg.kappa * cfg.gain * alpha

        # adjusted logits = base (re-read; weights may have moved) + accumulated suppression
        with torch.no_grad():
            adj_logits = actor(obs_t).squeeze(0).clone()
            sup_t = torch.as_tensor(self.sup, dtype=adj_logits.dtype, device=self.device)
            adj_logits = adj_logits + sup_t

        action = self._select(adj_logits)

        # diagnostics: per-action risk reduction (the SURROGATE metric; reported as diagnostic only)
        if diag["active"]:
            rr = R - compute_risk(cost, ttc, action)
            self.risk_red_sum += rr
            if rr > 0:
                self.lessrisky_hits += 1

        # ---- Recovery: confidence decay + Fisher-weighted weight pull toward W_0 ----
        self.sup *= cfg.rho_conf
        if cfg.weight_channel:
            with torch.no_grad():
                for name, p in actor.named_parameters():
                    if name not in self.w0:
                        continue
                    diff = self.w0[name].to(p.device) - p.data
                    fish = self.fisher.get(name)
                    if fish is None:
                        p.data.add_(cfg.eta_h * diff)
                    else:
                        p.data.add_(cfg.eta_h * fish.to(p.device) * diff)

        return action, diag


# ----------------------------------------------------------------------------- #
# Baseline interventions (each -> a single (CR, return) point).
# ----------------------------------------------------------------------------- #
def baseline_action(kind: str, logits: torch.Tensor, cost: float, ttc: float,
                    rng: np.random.Generator, ttc_k: float = 2.0,
                    temp: float = 2.0, base_decode: str = "sample") -> int:
    """Override rules act on the DEPLOYED policy action (base_decode), not on greedy.
    For stochastic deployment (the realistic regime), the un-overridden action is sampled,
    so prob_brake/ttc_brake trace a real frontier from the stochastic policy to always-brake.
    """
    def base():
        if base_decode == "greedy":
            return int(torch.argmax(logits).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())
    if kind == "noop_greedy":
        return int(torch.argmax(logits).item())
    if kind == "noop_sample":
        return int(torch.distributions.Categorical(logits=logits).sample().item())
    if kind == "ttc_brake":
        return 2 if ttc < ttc_k else base()                      # override to BRAKE when unsafe
    if kind == "prob_brake":
        return 2 if rng.random() < temp else base()              # tunable braking (p=temp)
    if kind == "action_mask":
        if ttc < ttc_k:                                          # block ACCELERATE, renormalize
            masked = logits.clone(); masked[1] = -1e9
            return int(torch.distributions.Categorical(logits=masked).sample().item()
                       if base_decode == "sample" else torch.argmax(masked).item())
        return base()
    if kind == "fixed_temp":
        return int(torch.distributions.Categorical(logits=logits / temp).sample().item())
    if kind == "cbf_lite":
        return 2 if (cost > 0.3 or ttc < ttc_k) else base()
    raise ValueError(kind)


# ----------------------------------------------------------------------------- #
# Bootstrap CI (separate RNG from experiment RNG; Rule: float64 accumulation).
# ----------------------------------------------------------------------------- #
def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "std": float(v.std()), "n": int(len(v))}
