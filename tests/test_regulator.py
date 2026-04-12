"""Unit tests for the regulator — catches the log_prob(mean) bug.

The critical test: REINFORCE requires log_prob of SAMPLED actions,
not log_prob of the distribution mean. log_prob(mean) is a constant
and produces zero gradient — the regulator never learns.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import sys
sys.path.insert(0, "D:/bci-2026")


def test_log_prob_varies_across_calls():
    """log_prob must vary across calls (different samples each time).
    If it doesn't, the REINFORCE gradient is zero."""
    mean = torch.tensor(0.5, requires_grad=True)
    std = torch.tensor(0.3)
    dist = Normal(mean, std)

    log_probs = []
    for _ in range(20):
        sample = dist.sample()
        lp = dist.log_prob(sample)
        log_probs.append(lp.item())

    # log_probs should vary because samples vary
    assert np.std(log_probs) > 0.01, \
        f"log_prob has zero variance ({np.std(log_probs):.6f}) — sampling is broken"


def test_log_prob_of_mean_is_constant():
    """Verify that log_prob(mean) is constant — this is the BUG pattern."""
    mean = torch.tensor(0.5)
    std = torch.tensor(0.3)
    dist = Normal(mean, std)

    # log_prob of the MEAN is always the same (this is what was broken)
    lp_of_mean = [dist.log_prob(mean).item() for _ in range(20)]
    assert np.std(lp_of_mean) < 1e-10, "log_prob(mean) should be constant"


def test_reinforce_gradient_nonzero():
    """REINFORCE gradient must be nonzero when reward varies."""
    param = nn.Parameter(torch.tensor(0.5))
    opt = torch.optim.Adam([param], lr=0.01)

    initial_val = param.item()

    for _ in range(50):
        dist = Normal(param, torch.tensor(0.3))
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        reward = sample.detach() - 0.5  # reward varies with sample

        loss = -(log_prob * reward)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert abs(param.item() - initial_val) > 0.001, \
        f"Parameter didn't move ({initial_val:.4f} -> {param.item():.4f}) — gradient is zero"


def test_reinforce_gradient_zero_with_mean():
    """Confirm that using log_prob(MEAN) gives zero gradient — the exact bug."""
    param = nn.Parameter(torch.tensor(0.5))
    opt = torch.optim.Adam([param], lr=0.01)

    initial_val = param.item()

    for _ in range(50):
        dist = Normal(param, torch.tensor(0.3))
        # BUG: using mean instead of sample
        log_prob = dist.log_prob(param)  # <-- THIS WAS THE BUG
        reward = torch.tensor(1.0)

        loss = -(log_prob * reward)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # With log_prob(mean), the gradient w.r.t. mean is always 0
    # because d/d_mu log N(mu; mu, sigma) = 0
    # So param should NOT move significantly
    # (it may drift slightly due to the -log(sigma*sqrt(2pi)) term)
    moved = abs(param.item() - initial_val)
    assert moved < 0.05, \
        f"Parameter moved {moved:.4f} with log_prob(mean) — unexpected"


def test_stats_regulator_learns():
    """Test that the FIXED SimpleRegulator actually learns from rewards."""
    from experiment_stats.run_multi_trial import SimpleRegulator

    device = "cpu"
    reg = SimpleRegulator(n_actions=4, n_groups=3, device=device)

    # Run 100 episodes of training with consistent positive reward
    lp_magnitudes = []
    for ep in range(100):
        # Simulate one FRA step
        logits = torch.randn(4)
        result = reg.act(
            fear=0.8, risk=0.7, logits=logits,
            cost=0.5, ttc=1.5, gpg=[0.1, 0.2, 0.3]
        )
        reg.store(result["lp"], 0.5)  # consistent positive reward
        reg.train_ep()
        lp_magnitudes.append(result["lp"].item())

    # The log_prob values should change over training (regulator is adapting)
    early = np.mean(lp_magnitudes[:10])
    late = np.mean(lp_magnitudes[-10:])
    changed = abs(late - early)
    assert changed > 0.01, \
        f"Regulator log_prob didn't change ({early:.4f} -> {late:.4f}) — not learning"


def test_stats_regulator_samples_differ():
    """Test that consecutive calls produce different outputs (sampling works)."""
    from experiment_stats.run_multi_trial import SimpleRegulator

    reg = SimpleRegulator(n_actions=4, n_groups=3, device="cpu")
    logits = torch.randn(4)

    outputs = []
    for _ in range(20):
        result = reg.act(fear=0.5, risk=0.5, logits=logits,
                         cost=0.3, ttc=3.0, gpg=[0.1, 0.1, 0.1])
        outputs.append(result["mag"])

    assert np.std(outputs) > 0.001, \
        f"Regulator outputs are constant (std={np.std(outputs):.6f}) — not sampling"


if __name__ == "__main__":
    tests = [
        test_log_prob_varies_across_calls,
        test_log_prob_of_mean_is_constant,
        test_reinforce_gradient_nonzero,
        test_reinforce_gradient_zero_with_mean,
        test_stats_regulator_learns,
        test_stats_regulator_samples_differ,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__} — {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
