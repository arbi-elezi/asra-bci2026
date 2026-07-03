"""Numerical verification of Corollary 2 (ranking-fidelity trajectory guarantee): for any base
policy and any critic COMONOTONE with the true cost-to-go (arbitrary values, same within-state
ranking), E_{pi_beta}[Q] <= E_{pi_0}[Q] for every gain; a single swapped pair breaks it."""
import numpy as np

rng = np.random.default_rng(0)
viol_como = viol_adv = cells = 0
for trial in range(2000):
    n = rng.integers(3, 8)
    logits = rng.normal(0, 2, n); p0 = np.exp(logits); p0 /= p0.sum()
    Q = rng.normal(0, 1, n)                                  # true Q^{pi0}(s,.)
    ranks = Q.argsort().argsort()
    Qhat = np.sort(rng.normal(0, 5, n))[ranks]               # comonotone, arbitrary values
    Qadv = Qhat.copy(); i, j = rng.choice(n, 2, replace=False)
    Qadv[i], Qadv[j] = Qadv[j], Qadv[i]                      # one discordant pair
    V0 = (p0 * Q).sum()
    for beta in [0.1, 0.5, 1, 3, 10]:
        cells += 1
        for Qc, which in [(Qhat, "como"), (Qadv, "adv")]:
            w = p0 * np.exp(-beta * (Qc - Qc.mean())); w /= w.sum()
            E = (w * Q).sum()
            if which == "como" and E > V0 + 1e-9: viol_como += 1
            if which == "adv" and E > V0 + 1e-9: viol_adv += 1
print(f"comonotone: {viol_como}/{cells} violations (theorem predicts 0)")
print(f"one swapped pair: {viol_adv}/{cells} violations (must be > 0: hypothesis load-bearing)")
assert viol_como == 0 and viol_adv > 0
print("Corollary 2 verified")
