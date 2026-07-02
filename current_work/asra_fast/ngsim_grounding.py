"""Real-trajectory grounding on NGSIM US-101 (public US DOT data): do the danger signal and the"""
from __future__ import annotations
import sys, json, argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.principled_asra import _load_cc
RES = Path("current_work/results_v3")
FT = 0.3048

# paper action indices: MAINTAIN=0, ACCELERATE=1, BRAKE=2, LANE_CHANGE=3


def load_ngsim():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob("data/ngsim/us101_*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for c in ("vehicle_id", "frame_id", "global_time", "preceding", "lane_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
    for c in ("v_vel", "v_acc", "space_headway", "time_headway", "local_y"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.dropna(subset=["v_vel", "space_headway"])
    return df


def build_frames(df):
    # preceding-vehicle speed lookup at same instant
    key = df.set_index(["vehicle_id", "global_time"])["v_vel"]
    prec_v = key.reindex(list(zip(df["preceding"], df["global_time"])))
    d = df.copy()
    d["v_prec"] = np.asarray(prec_v)
    d = d[(d["preceding"] > 0) & (d["space_headway"] > 0) & d["v_prec"].notna()]
    d["v_ego_ms"] = d["v_vel"] * FT
    d["v_prec_ms"] = d["v_prec"] * FT
    d["gap_m"] = d["space_headway"] * FT
    d["acc_ms2"] = d["v_acc"] * FT
    close = d["v_ego_ms"] - d["v_prec_ms"]
    ttc = np.where(close > 0.1, d["gap_m"] / np.maximum(close, 0.1), np.inf)
    d["ttc"] = ttc
    d["S"] = np.clip((2.0 - d["ttc"]) / 2.0, 0.0, 1.0)
    # imminent hard brake: min acc over the next 1 s (10 frames at 10 Hz) for the same vehicle
    d = d.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    g = d.groupby("vehicle_id")["acc_ms2"]
    fut_min = g.transform(lambda s: s[::-1].rolling(10, min_periods=1).min()[::-1].shift(-1))
    d["hard_brake_soon"] = (fut_min <= -3.0).astype(int)
    return d


def auc(scores, labels):
    from scipy.stats import rankdata
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def test_a(d):
    y = d["hard_brake_soon"].to_numpy()
    s = d["S"].to_numpy()
    inv_ttc = (1.0 / np.maximum(d["ttc"].to_numpy(), 0.1))
    fires = s > 0
    out = {"n_frames": int(len(d)), "base_rate": float(y.mean()),
           "auc_S": auc(s, y), "auc_invttc": auc(inv_ttc, y),
           "fire_rate": float(fires.mean()),
           "precision_at_S0": float(y[fires].mean() if fires.any() else 0.0),
           "recall_at_S0": float(y[fires].sum() / max(1, y.sum()))}
    print(f"[A] n={out['n_frames']} base={out['base_rate']:.3f} AUC(S)={out['auc_S']:.3f} "
          f"AUC(1/TTC)={out['auc_invttc']:.3f} P(S>0)={out['fire_rate']:.4f} "
          f"prec={out['precision_at_S0']:.3f} rec={out['recall_at_S0']:.3f}", flush=True)
    return out


def test_b(d, cc_path="checkpoints/cost_critic/full.pt", n_max=20000, seed=0):
    cc = _load_cc(cc_path)
    rng = np.random.default_rng(seed)
    # ALL danger frames (TTC<2 s) + a random benign sample
    danger_idx = np.flatnonzero((d["ttc"] < 2.0).to_numpy())
    benign_idx = rng.choice(np.flatnonzero((d["ttc"] >= 2.0).to_numpy()),
                            size=min(n_max, len(d)), replace=False)
    sub = d.iloc[np.concatenate([danger_idx, benign_idx])]
    obs = np.zeros((len(sub), 12), dtype=np.float32)
    obs[:, 0] = sub["local_y"] * FT          # ego x
    obs[:, 2] = sub["v_ego_ms"]              # ego vx
    obs[:, 4] = 1.0                          # cos_h
    obs[:, 6] = sub["gap_m"]                 # rel x of nearest
    obs[:, 8] = sub["v_prec_ms"] - sub["v_ego_ms"]  # rel vx (negative = closing)
    obs[:, 10] = 1.0
    with torch.no_grad():
        q = cc.predict_state(torch.from_numpy(obs), n_actions=4).numpy()
    brake_adv = q[:, 1] - q[:, 2]            # cost(ACCELERATE) - cost(BRAKE)
    danger = (sub["ttc"] < 2.0).to_numpy()
    ok_rank = (brake_adv > 0)
    from scipy.stats import spearmanr
    rho, p = spearmanr(sub["S"].to_numpy(), brake_adv)
    def bs_ci(x, B=2000):
        r = np.random.default_rng(1)
        m = [np.mean(x[r.integers(0, len(x), len(x))]) for _ in range(B)]
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    lo, hi = bs_ci(ok_rank[danger].astype(float)) if danger.sum() > 10 else (float("nan"),) * 2
    out = {"n": int(len(sub)), "n_danger": int(danger.sum()),
           "rank_acc_danger": float(ok_rank[danger].mean()) if danger.any() else float("nan"),
           "rank_acc_danger_ci": [lo, hi],
           "rank_acc_benign": float(ok_rank[~danger].mean()),
           "spearman_S_brakeadv": float(rho), "spearman_p": float(p)}
    print(f"[B] n={out['n']} danger={out['n_danger']} rank_acc(danger)={out['rank_acc_danger']:.3f} "
          f"CI[{lo:.3f},{hi:.3f}] rank_acc(benign)={out['rank_acc_benign']:.3f} "
          f"spearman(S, brake_adv)={rho:.3f} (p={p:.1e})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "ngsim_grounding.json")); a = ap.parse_args()
    d = build_frames(load_ngsim())
    out = {"A_salience_anticipates_braking": test_a(d), "B_critic_ranking_transfer": test_b(d)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)
