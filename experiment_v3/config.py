"""Experiment v3 configuration — typed dataclasses, no magic numbers."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Config for a single LLM model to test."""
    name: str                          # Human-readable name
    hf_id: str                         # HuggingFace model ID
    lora_rank: int = 8                 # LoRA rank
    lora_alpha: int = 16               # LoRA alpha
    lora_target_modules: list[str] | None = None  # Auto-detect if None
    action_head_hidden: int = 128      # Action head MLP hidden dim
    torch_dtype: str = "float16"       # Model dtype
    vram_estimate_gb: float = 1.0      # Approximate VRAM
    load_in_4bit: bool = False         # 4-bit quantization


# Models to test — ordered smallest to largest
# Local paths populated after download; HF IDs used as fallback
MODELS = [
    ModelConfig(
        name="SmolLM2-135M",
        hf_id="D:/bci-2026/models/SmolLM2-135M",
        lora_rank=8, lora_alpha=16,
        action_head_hidden=64,
        vram_estimate_gb=0.3,
    ),
    ModelConfig(
        name="SmolLM2-360M",
        hf_id="D:/bci-2026/models/SmolLM2-360M",
        lora_rank=8, lora_alpha=16,
        action_head_hidden=128,
        vram_estimate_gb=0.7,
    ),
    ModelConfig(
        name="Qwen2.5-0.5B",
        hf_id="D:/bci-2026/models/Qwen2.5-0.5B",
        lora_rank=16, lora_alpha=32,
        action_head_hidden=128,
        vram_estimate_gb=1.0,
    ),
    ModelConfig(
        name="TinyLlama-1.1B",
        hf_id="D:/bci-2026/models/TinyLlama-1.1B",
        lora_rank=16, lora_alpha=32,
        action_head_hidden=128,
        vram_estimate_gb=2.2,
    ),
    ModelConfig(
        name="Qwen7B-4bit",
        hf_id="D:/csharp-llm/models/qwen_7b_base",
        lora_rank=16, lora_alpha=32,
        action_head_hidden=256,
        torch_dtype="float16",
        vram_estimate_gb=6.0,
        load_in_4bit=True,
    ),
]

# MLP baselines (no LLM, for comparison)
MLP_CONFIGS = [
    {"name": "MLP-small", "layers": [12, 128, 128, 4], "desc": "2-layer MLP (v1 baseline)"},
    {"name": "MLP-large", "layers": [12, 256, 256, 128, 4], "desc": "3-layer deep MLP"},
]


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # RL training (action head + LoRA fine-tuning)
    ppo_total_steps: int = 100_000     # Total environment steps
    ppo_lr: float = 3e-4               # Learning rate
    ppo_batch_size: int = 64           # Minibatch size
    ppo_n_epochs: int = 4              # PPO epochs per rollout
    ppo_n_steps: int = 512             # Steps per rollout
    ppo_gamma: float = 0.99            # Discount factor
    ppo_gae_lambda: float = 0.95       # GAE lambda
    ppo_clip_range: float = 0.2        # PPO clip
    ppo_ent_coef: float = 0.01         # Entropy bonus
    ppo_vf_coef: float = 0.5           # Value loss coefficient
    ppo_max_grad_norm: float = 0.5     # Gradient clipping

    # Cost critic
    cost_critic_epochs: int = 200      # Training epochs
    cost_critic_lr: float = 1e-3
    cost_critic_hidden: int = 128      # Hidden dim (bigger than v1's 64)
    cost_critic_layers: int = 3        # Depth (bigger than v1's 2)

    # D_ref
    d_ref_size: int = 10_000           # 10x more than v1's 1000

    # Fisher
    fisher_samples: int = 2000         # Samples for Fisher estimation
    fisher_regularization: float = 1e-3  # Ensures f_min > 0 meaningfully

    # Hyperparameter search
    hp_search_seeds: int = 200         # Validation seeds for HP search
    hp_search_configs: int = 50        # Number of HP configs to try

    # Checkpointing
    checkpoint_every: int = 1000       # Steps between checkpoints
    eval_every: int = 2000             # Steps between evaluations


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    n_experiment_seeds: int = 1000     # Paper spec
    n_stress_seeds: int = 500          # Paper spec for C8
    n_validation_seeds: int = 200      # For hyperparameter search
    vehicles_count: int = 15           # highway-env vehicles
    max_episode_steps: int = 500       # Episode length
    bootstrap_samples: int = 10_000    # Bootstrap CI resamples
    device: str = "cuda"
    output_base: Path = field(default_factory=lambda: Path("experiment_v3"))

    # FRA hyperparameter ranges (Table 4)
    eta_f_range: tuple[float, float] = (1e-4, 1e-1)
    eta_h_range: tuple[float, float] = (1e-6, 1e-2)
    eta_bc_range: tuple[float, float] = (1e-5, 1e-3)
    gamma_f_range: tuple[float, float] = (0.0, 0.9)
    beta_range: tuple[float, float] = (0.0, 2.0)
    tau_range: tuple[float, float] = (0.2, 0.8)
    k_range: tuple[float, float] = (1.0, 20.0)
