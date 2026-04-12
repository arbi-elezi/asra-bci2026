"""LLM Decision Maker — Small open-source LLM with LoRA perturbation head.

Architecture:
  - Base: TinyLlama-1.1B (or configurable)
  - Action head: LoRA adapters → 4-action softmax
  - W_0: Initial LoRA weights (FROZEN snapshot)
  - W_t: Current LoRA weights (perturbed by FRA fear mechanism)

Scientific justification:
  - LoRA perturbation is mathematically equivalent to full perturbation in a
    low-rank subspace: W_t = W_0 + B·A
  - Proposition 1 bounds ||B·A||_F directly
  - Fisher diagonal computed over LoRA params only (~1M vs 1B)
  - FHR restoring force pulls B·A → 0 (i.e., W_t → W_0)

The LLM receives a text description of the driving state and outputs
action logits over the 4 discrete actions.
"""

from __future__ import annotations

import hashlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class LLMDriverConfig:
    """Configuration for the LLM decision maker."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    n_actions: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # 0 for determinism during eval
    lora_target_modules: list[str] | None = None  # Auto-detect
    device: str = "cuda"
    torch_dtype: str = "float16"
    max_new_tokens: int = 4
    # Action head: project last hidden state → n_actions
    action_head_hidden: int = 64


class ActionHead(nn.Module):
    """Maps LLM hidden state → action logits.

    Small MLP: hidden_dim → 64 → 4 (softmax applied externally).
    This is PART of the perturbable parameters.
    """

    def __init__(self, hidden_dim: int, n_actions: int = 4, mid_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, n_actions),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: [batch, hidden_dim] → logits: [batch, n_actions]."""
        return self.net(hidden_state)


class LLMDriver:
    """Small open-source LLM as driving decision maker.

    The LLM's LoRA weights + action head are the perturbable parameter set.
    FRA perturbs these weights via fear-triggered DR gradients and
    FHR homeostatic restoring force.
    """

    def __init__(self, config: LLMDriverConfig) -> None:
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # ── Load base LLM ──
        print(f"Loading {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )

        # ── Apply LoRA ──
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,  # Auto-detect if None
        )
        self.model = get_peft_model(self.base_model, lora_config)
        print(f"LoRA params: {self.model.print_trainable_parameters()}")

        # ── Action head ──
        hidden_dim = self.model.config.hidden_size
        self.action_head = ActionHead(
            hidden_dim, config.n_actions, config.action_head_hidden
        ).to(self.device).to(self.dtype)

        # ── W_0 snapshot (FROZEN — NEVER modified) ──
        self.w0_lora = self._snapshot_lora_params()
        self.w0_action_head = {
            k: v.clone().detach() for k, v in self.action_head.state_dict().items()
        }
        self._w0_hash = self._compute_hash()

    def get_action_logits(self, state_text: str) -> torch.Tensor:
        """Convert state text → action logits via LLM + action head.

        Pipeline:
          1. Tokenize state description
          2. Forward through LLM (with LoRA)
          3. Extract last hidden state
          4. Project through action head → 4 logits

        Args:
            state_text: Natural language driving state description.

        Returns:
            logits: [n_actions] tensor.
        """
        inputs = self.tokenizer(
            state_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
            # Last hidden state at the last token position
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

        # Action head (this CAN have gradients — it's perturbable)
        logits = self.action_head(last_hidden.to(self.dtype))
        return logits.squeeze(0)  # [n_actions]

    def get_action(self, state_text: str) -> tuple[int, torch.Tensor]:
        """Get action from LLM policy.

        Returns:
            (action_index, action_probabilities)
        """
        logits = self.get_action_logits(state_text)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs.float(), 1).item()
        return action, probs

    def get_action_logits_from_obs(self, obs: np.ndarray, state_text_fn) -> torch.Tensor:
        """Get logits from numeric obs via text conversion.

        Args:
            obs: R^12 observation.
            state_text_fn: Function that converts obs → text.

        Returns:
            logits: [n_actions] tensor.
        """
        text = state_text_fn(obs)
        return self.get_action_logits(text)

    # ── LoRA Parameter Access (for FRA perturbation) ──

    def get_perturbable_params(self) -> list[tuple[str, nn.Parameter]]:
        """Return all perturbable parameters (LoRA + action head).

        These are the params that DR and FHR operate on.
        W_0 is the snapshot; these are the live W_t values.
        """
        params = []

        # LoRA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only LoRA params have requires_grad=True
                params.append((f"lora.{name}", param))

        # Action head parameters
        for name, param in self.action_head.named_parameters():
            params.append((f"action_head.{name}", param))

        return params

    def get_w0_params(self) -> dict[str, torch.Tensor]:
        """Return the frozen W_0 snapshot."""
        w0 = {}
        for k, v in self.w0_lora.items():
            w0[f"lora.{k}"] = v
        for k, v in self.w0_action_head.items():
            w0[f"action_head.{k}"] = v
        return w0

    def _snapshot_lora_params(self) -> dict[str, torch.Tensor]:
        """Take a snapshot of current LoRA parameters."""
        snapshot = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                snapshot[name] = param.data.clone().detach()
        return snapshot

    def restore_to_w0(self) -> None:
        """Restore all perturbable params to W_0 snapshot."""
        # Restore LoRA
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.w0_lora:
                param.data.copy_(self.w0_lora[name])

        # Restore action head
        self.action_head.load_state_dict(
            {k: v.clone() for k, v in self.w0_action_head.items()}
        )

    def _compute_hash(self) -> str:
        """SHA-256 hash of W_0 for reproducibility verification."""
        hasher = hashlib.sha256()
        for v in self.w0_lora.values():
            hasher.update(v.cpu().float().numpy().tobytes())
        for v in self.w0_action_head.values():
            hasher.update(v.cpu().float().numpy().tobytes())
        return hasher.hexdigest()

    def verify_w0_intact(self) -> bool:
        """Verify W_0 snapshot hasn't been corrupted."""
        return self._compute_hash() == self._w0_hash

    def get_w0_hash(self) -> str:
        return self._w0_hash

    def count_perturbable_params(self) -> int:
        """Count total perturbable parameters."""
        return sum(p.numel() for _, p in self.get_perturbable_params())
