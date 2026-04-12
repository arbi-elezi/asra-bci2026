"""LLM-based driving policy with LoRA adapters.

This is the REAL implementation — not a toy MLP.
The LLM processes state text → hidden representation → action head → logits.
LoRA adapters + action head are the perturbable parameter set for FRA.
"""
from __future__ import annotations

import hashlib
import os
import ssl
from typing import Any

# Fix SSL for environments with missing/untrusted root CA
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from .config import ModelConfig


class ActionHead(nn.Module):
    """Projects LLM hidden state → action logits.

    Two-layer MLP: hidden_dim → action_head_hidden → n_actions.
    This is PART of the perturbable parameters.
    """
    def __init__(self, hidden_dim: int, n_actions: int = 4, mid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, n_actions),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class LLMPolicy:
    """LLM driving policy with LoRA + action head.

    Architecture:
      1. State text → tokenizer → tokens
      2. Tokens → LLM (frozen base + trainable LoRA) → hidden states
      3. Last hidden state → action head → 4 logits
      4. Logits → action distribution → action

    The perturbable set is: LoRA adapters + action head weights.
    The base LLM weights are NEVER modified.
    """

    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.cfg = config
        self.device = torch.device(device)

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # Load base LLM
        print(f"  Loading {config.hf_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": self.device},
        }
        if config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.base_model = AutoModelForCausalLM.from_pretrained(config.hf_id, **load_kwargs)
        if config.load_in_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,  # Deterministic during eval
            target_modules=config.lora_target_modules,
        )
        self.model = get_peft_model(self.base_model, lora_config)

        # Action head — ALWAYS fp32 for numerical stability during RL training
        # LLM outputs fp16 hidden states, but action head computes in fp32
        # to avoid NaN in log_prob/softmax during PPO updates
        hidden_dim = self.model.config.hidden_size
        self.action_head = ActionHead(
            hidden_dim, n_actions=4, mid_dim=config.action_head_hidden
        ).to(self.device).float()  # fp32 always

        # Count params
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.action_head.parameters())
        self.n_perturbable = lora_params + head_params
        print(f"  LoRA params: {lora_params:,}, Action head: {head_params:,}, Total perturbable: {self.n_perturbable:,}")

        # W_0 snapshot (taken after training)
        self._w0: dict[str, torch.Tensor] | None = None
        self._w0_hash: str | None = None

    def get_logits(self, state_text: str) -> torch.Tensor:
        """State text → action logits [4]."""
        inputs = self.tokenizer(
            state_text, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]
        # Cast to fp32 for action head — prevents NaN in softmax/log_prob
        logits = self.action_head(last_hidden.float())
        return logits.squeeze(0)  # [4]

    def get_logits_from_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Numeric observation → action logits."""
        text = self._obs_to_text(obs)
        return self.get_logits(text)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Get action from observation."""
        with torch.no_grad():
            logits = self.get_logits_from_obs(obs)
            if deterministic:
                return logits.argmax().item()
            return torch.distributions.Categorical(logits=logits).sample().item()

    def get_perturbable_params(self) -> list[tuple[str, nn.Parameter]]:
        """Return all perturbable parameters (LoRA + action head)."""
        params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                params.append((f"lora.{name}", p))
        for name, p in self.action_head.named_parameters():
            params.append((f"head.{name}", p))
        return params

    def snapshot_w0(self) -> None:
        """Take W_0 snapshot. Call AFTER training, BEFORE experiments."""
        self._w0 = {name: p.data.clone().detach() for name, p in self.get_perturbable_params()}
        self._w0_hash = self._compute_hash()
        print(f"  W_0 frozen. Hash: {self._w0_hash[:16]}... ({self.n_perturbable:,} params)")

    def restore_w0(self) -> None:
        """Restore all perturbable params to W_0."""
        assert self._w0 is not None, "Call snapshot_w0() first"
        for name, p in self.get_perturbable_params():
            if name in self._w0:
                p.data.copy_(self._w0[name])

    def verify_w0(self) -> bool:
        """Verify W_0 hasn't been corrupted."""
        return self._compute_hash() == self._w0_hash

    def get_w0(self) -> dict[str, torch.Tensor]:
        assert self._w0 is not None
        return self._w0

    def get_w0_hash(self) -> str:
        return self._w0_hash or ""

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        for _, p in self.get_perturbable_params():
            h.update(p.data.cpu().float().numpy().tobytes())
        return h.hexdigest()

    @staticmethod
    def _obs_to_text(obs: np.ndarray) -> str:
        """Convert R^12 observation to natural language."""
        ego_vx = obs[2] if len(obs) > 2 else 25.0
        rel_x = obs[6] if len(obs) > 6 else 100.0
        rel_vx = obs[8] if len(obs) > 8 else 0.0

        ttc = (-rel_x / rel_vx) if (rel_x > 0 and rel_vx < 0) else 10.0
        ttc = min(max(ttc, 0), 10.0)
        cost = max(0.0, (2.0 - ttc) / 2.0)

        speed_kmh = ego_vx * 3.6
        gap = rel_x
        closing = abs(rel_vx * 3.6)

        if cost > 0.7:
            danger = "CRITICAL"
        elif cost > 0.3:
            danger = "HIGH"
        elif cost > 0:
            danger = "MODERATE"
        else:
            danger = "SAFE"

        return (
            f"Highway driving: speed {speed_kmh:.0f}km/h, "
            f"gap {gap:.0f}m, closing {closing:.0f}km/h, "
            f"TTC {ttc:.1f}s, threat {danger}. "
            f"Actions: 0=maintain 1=accelerate 2=brake 3=lane-change"
        )

    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items() if "lora" in k},
            "action_head_state": self.action_head.state_dict(),
            "w0": {k: v.cpu() for k, v in self._w0.items()} if self._w0 else None,
            "w0_hash": self._w0_hash,
            "config": self.cfg,
        }, path)

    def load(self, path: str) -> None:
        """Load model state."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        # Load LoRA weights
        current = self.model.state_dict()
        for k, v in data["model_state"].items():
            if k in current:
                current[k] = v.to(self.device)
        self.model.load_state_dict(current, strict=False)
        # Load action head
        self.action_head.load_state_dict(
            {k: v.to(self.device) for k, v in data["action_head_state"].items()}
        )
        # Load W_0
        if data.get("w0"):
            self._w0 = {k: v.to(self.device) for k, v in data["w0"].items()}
            self._w0_hash = data["w0_hash"]
