"""FRA Demo Video Capture — Slow-motion simulation with real-time perturbation stats.

Captures two synchronized panels:
  LEFT:  Highway-env simulation (bird's-eye or top-down view)
  RIGHT: Real-time perturbation dashboard showing:
         - Weight Deviation Norm (M3) — bar + time series
         - Gradient Norm (M13) — bar with A1 bound line
         - Fear Signal F_t — pipeline: raw → TD → FMS → final
         - SCL mixing coefficient α_t
         - LoRA weight heatmap (per-layer perturbation magnitude)
         - Recovery timer (steps since last spike)
         - Cost signal c_t with TTC
         - Episode stats (CR so far, current reward)

Output: MP4 at configurable FPS (default 5 for slow-mo research viewing).

Usage:
  python demo_capture.py --output demo.mp4 --episodes 5 --fps 5
  python demo_capture.py --output demo.mp4 --episodes 1 --fps 2 --slow
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.environment.highway_wrapper import HighwayFRAEnv


# ── Color palette ──────────────────────────────────────────────────────────

COL_BG = (20, 20, 30)
COL_TEXT = (220, 220, 230)
COL_FEAR_LOW = (60, 180, 60)       # Green
COL_FEAR_MED = (220, 180, 30)      # Yellow
COL_FEAR_HIGH = (220, 60, 60)      # Red
COL_WDN = (80, 140, 255)           # Blue
COL_GRAD = (200, 100, 255)         # Purple
COL_SCL = (255, 180, 50)           # Orange
COL_COST = (255, 80, 80)           # Red
COL_BOUND = (100, 255, 100)        # Green — A1 bound line
COL_RECOVERY = (100, 200, 200)     # Cyan
COL_SPIKE = (255, 50, 50)          # Bright red — fear spike marker


def fear_color(f: float) -> tuple[int, int, int]:
    """Interpolate fear color: green → yellow → red."""
    if f < 0.3:
        t = f / 0.3
        return (
            int(COL_FEAR_LOW[0] + t * (COL_FEAR_MED[0] - COL_FEAR_LOW[0])),
            int(COL_FEAR_LOW[1] + t * (COL_FEAR_MED[1] - COL_FEAR_LOW[1])),
            int(COL_FEAR_LOW[2] + t * (COL_FEAR_MED[2] - COL_FEAR_LOW[2])),
        )
    else:
        t = min(1.0, (f - 0.3) / 0.7)
        return (
            int(COL_FEAR_MED[0] + t * (COL_FEAR_HIGH[0] - COL_FEAR_MED[0])),
            int(COL_FEAR_MED[1] + t * (COL_FEAR_HIGH[1] - COL_FEAR_MED[1])),
            int(COL_FEAR_MED[2] + t * (COL_FEAR_HIGH[2] - COL_FEAR_MED[2])),
        )


class PerturbationDashboard:
    """Renders the right-panel dashboard as a matplotlib figure → numpy RGB."""

    def __init__(self, width: int = 640, height: int = 720, history_len: int = 200):
        self.w = width
        self.h = height
        self.history_len = history_len
        self.dpi = 100

        # Rolling histories
        self.wdn_history = deque(maxlen=history_len)
        self.grad_history = deque(maxlen=history_len)
        self.fear_raw_history = deque(maxlen=history_len)
        self.fear_final_history = deque(maxlen=history_len)
        self.alpha_history = deque(maxlen=history_len)
        self.cost_history = deque(maxlen=history_len)
        self.ttc_history = deque(maxlen=history_len)

        # LoRA layer perturbation magnitudes (simulated for demo)
        self.lora_layers = 8  # Typical TinyLlama LoRA layer count
        self.layer_perturbations = np.zeros(self.lora_layers)

        # Spike tracking
        self.steps_since_spike = 0
        self.spike_steps = []  # Step indices of fear spikes

    def update(self, step_info: dict) -> None:
        """Ingest one timestep's worth of data."""
        wdn = step_info.get("wdn", 0.0)
        grad = step_info.get("grad_norm", 0.0)
        fear_raw = step_info.get("fear_raw", 0.0)
        fear_final = step_info.get("fear_final", 0.0)
        alpha = step_info.get("alpha", 0.0)
        cost = step_info.get("cost", 0.0)
        ttc = step_info.get("ttc", 10.0)

        self.wdn_history.append(wdn)
        self.grad_history.append(grad)
        self.fear_raw_history.append(fear_raw)
        self.fear_final_history.append(fear_final)
        self.alpha_history.append(alpha)
        self.cost_history.append(cost)
        self.ttc_history.append(ttc)

        # Track spikes
        if fear_final > 0.3:
            self.steps_since_spike = 0
            self.spike_steps.append(len(self.wdn_history))
        else:
            self.steps_since_spike += 1

        # Simulate LoRA layer perturbation pattern (wave decay)
        if wdn > 0:
            for i in range(self.lora_layers):
                wave_factor = 0.9 ** i  # Wave decay
                self.layer_perturbations[i] = wdn * wave_factor * (0.8 + 0.4 * np.random.random())
        else:
            self.layer_perturbations *= 0.95  # Slow decay

    def render(self, g_max: float = 10.0, bound: float = 1.0, episode_stats: dict | None = None) -> np.ndarray:
        """Render the full dashboard as an RGB numpy array."""
        fig_w = self.w / self.dpi
        fig_h = self.h / self.dpi
        fig = Figure(figsize=(fig_w, fig_h), dpi=self.dpi, facecolor="#14141e")
        canvas = FigureCanvasAgg(fig)

        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35,
                               left=0.12, right=0.95, top=0.95, bottom=0.05)

        # ── 1. WDN time series (top left) ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#1a1a2e")
        wdn_arr = np.array(self.wdn_history) if self.wdn_history else np.zeros(1)
        ax1.plot(wdn_arr, color="#508cff", linewidth=1.2, label="WDN")
        ax1.axhline(y=bound, color="#64ff64", linewidth=0.8, linestyle="--", alpha=0.7, label=f"bound={bound:.2f}")
        # Mark spike moments
        for sp in self.spike_steps:
            if sp < len(wdn_arr):
                ax1.axvline(x=sp, color="#ff3232", linewidth=0.4, alpha=0.3)
        ax1.set_title("M3: Weight Deviation Norm", color="white", fontsize=8, fontweight="bold")
        ax1.set_ylabel("||W_t - W_0||_F", color="white", fontsize=7)
        ax1.tick_params(colors="white", labelsize=6)
        ax1.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
        ax1.set_xlim(0, max(len(wdn_arr), 10))

        # ── 2. Gradient norm (top right) ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#1a1a2e")
        grad_arr = np.array(self.grad_history) if self.grad_history else np.zeros(1)
        ax2.plot(grad_arr, color="#c864ff", linewidth=1.2)
        ax2.axhline(y=g_max, color="#64ff64", linewidth=0.8, linestyle="--", alpha=0.7, label=f"G_max={g_max:.1f}")
        violations = np.sum(grad_arr > g_max)
        viol_frac = violations / max(len(grad_arr), 1)
        ax2.set_title(f"M13: Gradient Norm (A1 viol: {viol_frac:.3f})", color="white", fontsize=8, fontweight="bold")
        ax2.set_ylabel("||G_t^DR||_F", color="white", fontsize=7)
        ax2.tick_params(colors="white", labelsize=6)
        ax2.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")

        # ── 3. Fear pipeline (middle left) ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor("#1a1a2e")
        fr = np.array(self.fear_raw_history) if self.fear_raw_history else np.zeros(1)
        ff = np.array(self.fear_final_history) if self.fear_final_history else np.zeros(1)
        ax3.fill_between(range(len(ff)), ff, alpha=0.3, color="#ff4444")
        ax3.plot(fr, color="#ff8888", linewidth=0.8, alpha=0.6, label="raw")
        ax3.plot(ff, color="#ff4444", linewidth=1.2, label="final")
        ax3.axhline(y=0.3, color="yellow", linewidth=0.5, linestyle=":", alpha=0.5)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_title("Fear Signal F_t", color="white", fontsize=8, fontweight="bold")
        ax3.tick_params(colors="white", labelsize=6)
        ax3.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")

        # ── 4. SCL + Cost (middle right) ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor("#1a1a2e")
        al = np.array(self.alpha_history) if self.alpha_history else np.zeros(1)
        co = np.array(self.cost_history) if self.cost_history else np.zeros(1)
        ax4.plot(co, color="#ff5050", linewidth=1.0, label="cost c_t")
        ax4.plot(al, color="#ffb432", linewidth=1.0, label="SCL α_t")
        ax4.set_ylim(-0.05, 1.05)
        ax4.set_title("Cost & SCL Mixing", color="white", fontsize=8, fontweight="bold")
        ax4.tick_params(colors="white", labelsize=6)
        ax4.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")

        # ── 5. LoRA weight heatmap (bottom left) ──
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_facecolor("#1a1a2e")
        heatmap = self.layer_perturbations.reshape(1, -1)
        im = ax5.imshow(heatmap, aspect="auto", cmap="inferno", vmin=0,
                        vmax=max(0.01, self.layer_perturbations.max() * 1.2))
        ax5.set_yticks([])
        ax5.set_xticks(range(self.lora_layers))
        ax5.set_xticklabels([f"L{i}" for i in range(self.lora_layers)], color="white", fontsize=7)
        ax5.set_title("LoRA Layer Perturbation Magnitude (wave propagation →)", color="white", fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax5, fraction=0.02, pad=0.04)

        # ── 6. TTC time series (row 3 left) ──
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.set_facecolor("#1a1a2e")
        ttc_arr = np.array(self.ttc_history) if self.ttc_history else np.ones(1) * 10
        ax6.plot(ttc_arr, color="#64c8ff", linewidth=1.0)
        ax6.axhline(y=2.0, color="#ff5050", linewidth=0.8, linestyle="--", alpha=0.7, label="TTC=2s (danger)")
        ax6.set_ylim(-0.5, 10.5)
        ax6.set_title("TTC (Time-to-Collision)", color="white", fontsize=8, fontweight="bold")
        ax6.set_ylabel("seconds", color="white", fontsize=7)
        ax6.tick_params(colors="white", labelsize=6)
        ax6.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")

        # ── 7. Episode stats text (row 3 right) ──
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.set_facecolor("#1a1a2e")
        ax7.axis("off")
        stats = episode_stats or {}
        lines = [
            f"Episode: {stats.get('episode', 0)}",
            f"Step: {stats.get('step', 0)}",
            f"CR so far: {stats.get('cr', 0.0):.3f}",
            f"Reward: {stats.get('reward', 0.0):.1f}",
            f"Recovery: {self.steps_since_spike} steps",
            f"Spikes: {len(self.spike_steps)}",
            f"WDN now: {wdn_arr[-1]:.4f}" if len(wdn_arr) > 0 else "",
        ]
        text = "\n".join(lines)
        ax7.text(0.1, 0.9, text, transform=ax7.transAxes, color="white",
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a3e", edgecolor="gray"))

        # ── 8. Recovery gauge (bottom) ──
        ax8 = fig.add_subplot(gs[4, :])
        ax8.set_facecolor("#1a1a2e")
        # Show temporal gradient re-ascent multiplier
        gamma = 0.02
        max_mult = 10.0
        recovery_mult = min(1.0 + gamma * self.steps_since_spike, max_mult)
        bar_width = recovery_mult / max_mult
        ax8.barh(0, bar_width, height=0.6, color="#64c8c8", alpha=0.8)
        ax8.barh(0, 1.0, height=0.6, color="none", edgecolor="gray", linewidth=0.5)
        ax8.set_xlim(0, 1.0)
        ax8.set_yticks([])
        ax8.set_title(f"FHR Recovery Force: {recovery_mult:.1f}x base (temporal gradient re-ascent)", color="white", fontsize=8, fontweight="bold")
        ax8.tick_params(colors="white", labelsize=6)

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self.h, self.w, 4)[:, :, :3]  # RGBA → RGB
        plt.close(fig)

        return buf

    def reset(self) -> None:
        """Clear all histories for new episode."""
        self.wdn_history.clear()
        self.grad_history.clear()
        self.fear_raw_history.clear()
        self.fear_final_history.clear()
        self.alpha_history.clear()
        self.cost_history.clear()
        self.ttc_history.clear()
        self.layer_perturbations = np.zeros(self.lora_layers)
        self.steps_since_spike = 0
        self.spike_steps = []


def render_sim_frame(env: HighwayFRAEnv, obs: np.ndarray, info: dict,
                     width: int = 640, height: int = 720) -> np.ndarray:
    """Render the simulation panel (left side).

    Uses highway-env's built-in renderer if available, otherwise
    draws a custom top-down view with obstacle class coloring.
    """
    # Try highway-env's native rendering
    try:
        raw_frame = env._inner.render()
        if raw_frame is not None and isinstance(raw_frame, np.ndarray):
            # Resize to target dimensions
            frame = cv2.resize(raw_frame, (width, height - 120))
            # Add dark panel for text overlay at bottom
            panel = np.full((120, width, 3), COL_BG, dtype=np.uint8)
            frame = np.vstack([frame, panel])

            # Overlay text
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)
            step = info.get("step", 0)
            obs_class = info.get("obstacle_class", -1)
            cls_names = {0: "SLOW", 1: "FAST", 2: "STATIONARY", -1: "NONE"}

            texts = [
                f"Step {step:>4d}  |  TTC: {ttc:>5.1f}s  |  Cost: {cost:.3f}",
                f"Nearest: {cls_names.get(obs_class, '?')}  |  "
                f"Speed: {obs[2]*3.6:.0f} km/h  |  Gap: {obs[6]:.1f}m",
            ]

            # Color cost bar
            cost_bar_w = int(cost * (width - 40))
            bar_y = height - 50
            cv2.rectangle(frame, (20, bar_y), (20 + cost_bar_w, bar_y + 15),
                          (0, 0, int(255 * cost)), -1)
            cv2.rectangle(frame, (20, bar_y), (width - 20, bar_y + 15),
                          (80, 80, 80), 1)

            for i, text in enumerate(texts):
                y = height - 100 + i * 22
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, COL_TEXT, 1, cv2.LINE_AA)

            return frame
    except Exception:
        pass

    # Fallback: custom top-down rendering
    frame = np.full((height, width, 3), COL_BG, dtype=np.uint8)

    # Road
    road_top = 100
    road_bot = height - 200
    cv2.rectangle(frame, (0, road_top), (width, road_bot), (60, 60, 70), -1)

    # Lane markings
    n_lanes = 3
    for i in range(n_lanes + 1):
        y = road_top + int(i * (road_bot - road_top) / n_lanes)
        for x in range(0, width, 30):
            cv2.line(frame, (x, y), (x + 15, y), (200, 200, 180), 1)

    # Ego vehicle (green)
    ego_y = road_top + int(1.5 * (road_bot - road_top) / n_lanes)
    ego_x = width // 4
    cv2.rectangle(frame, (ego_x - 15, ego_y - 8), (ego_x + 15, ego_y + 8),
                  (0, 255, 0), -1)
    cv2.putText(frame, "EGO", (ego_x - 12, ego_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 0), 1)

    # Nearest obstacle (colored by class)
    if obs[6] != 0 or obs[7] != 0:
        obs_x = ego_x + int(obs[6] * 2)  # Scale gap to pixels
        obs_y = ego_y + int(obs[7] * 20)
        obs_class = info.get("obstacle_class", -1)
        colors = {0: (100, 100, 255), 1: (255, 100, 100), 2: (200, 200, 0), -1: (128, 128, 128)}
        color = colors.get(obs_class, (128, 128, 128))
        cv2.rectangle(frame, (obs_x - 12, obs_y - 7), (obs_x + 12, obs_y + 7), color, -1)

    # Info text at bottom
    cost = info.get("cost", 0.0)
    ttc = info.get("ttc", 10.0)
    texts = [
        f"Step {info.get('step', 0):>4d}",
        f"TTC: {ttc:.1f}s  Cost: {cost:.3f}",
        f"Speed: {obs[2]*3.6:.0f} km/h",
    ]
    for i, text in enumerate(texts):
        y = height - 160 + i * 20
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COL_TEXT, 1, cv2.LINE_AA)

    return frame


def run_demo_capture(
    output_path: str = "demo.mp4",
    n_episodes: int = 3,
    fps: int = 5,
    seed: int = 42,
    sim_width: int = 640,
    dash_width: int = 640,
    frame_height: int = 720,
    g_max: float = 10.0,
    prop1_bound: float = 1.0,
) -> None:
    """Capture demo video with simulation + perturbation dashboard.

    Args:
        output_path: Output MP4 path.
        n_episodes: Number of episodes to record.
        fps: Output video FPS (low = slow-motion for research viewing).
        seed: Starting seed.
        sim_width: Simulation panel width.
        dash_width: Dashboard panel width.
        frame_height: Total frame height.
        g_max: A1 gradient norm bound for M13 display.
        prop1_bound: Proposition 1 WDN bound for display.
    """
    total_width = sim_width + dash_width

    # Initialize environment
    env = HighwayFRAEnv(render_mode="rgb_array", seed=seed)

    # Initialize dashboard
    dashboard = PerturbationDashboard(width=dash_width, height=frame_height)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, frame_height))

    if not writer.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, frame_height))

    print(f"Recording demo to {output_path}")
    print(f"Resolution: {total_width}x{frame_height} @ {fps} FPS")
    print(f"Episodes: {n_episodes}")

    total_frames = 0
    total_collisions = 0
    total_episodes_done = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        dashboard.reset()

        episode_reward = 0.0
        episode_collision = False

        print(f"\n  Episode {ep + 1}/{n_episodes} (seed={seed + ep})")

        for t in range(500):
            # Simulate FRA step data (when running without LLM, use synthetic signals)
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            # Synthetic perturbation signals for demo
            # In production, these come from FRAEngine.step()
            fear_raw = cost * 0.8 + np.random.random() * 0.1
            fear_raw = max(0.0, min(1.0, fear_raw))
            # TD smoothing simulation
            fear_td = fear_raw * 0.7 + (dashboard.fear_final_history[-1] if dashboard.fear_final_history else 0) * 0.3
            fear_final = max(0.0, min(1.0, fear_td))

            wdn = fear_final * 0.3 * (1.0 + 0.2 * np.sin(t * 0.1))
            if wdn < 0.01:
                wdn *= 0.9  # Decay
            grad_norm = fear_final * g_max * 0.6 * (1.0 + 0.3 * np.random.random())
            alpha = 1.0 / (1.0 + np.exp(-10 * (fear_final - 0.5)))

            step_info = {
                "wdn": wdn,
                "grad_norm": grad_norm,
                "fear_raw": fear_raw,
                "fear_final": fear_final,
                "alpha": alpha,
                "cost": cost,
                "ttc": ttc,
            }

            dashboard.update(step_info)

            # Render both panels
            sim_frame = render_sim_frame(env, obs, info, sim_width, frame_height)
            dash_frame = dashboard.render(
                g_max=g_max,
                bound=prop1_bound,
                episode_stats={
                    "episode": ep + 1,
                    "step": t,
                    "cr": total_collisions / max(total_episodes_done, 1),
                    "reward": episode_reward,
                },
            )

            # Combine panels side by side
            combined = np.hstack([sim_frame, dash_frame])

            # Separator line
            cv2.line(combined, (sim_width, 0), (sim_width, frame_height), (60, 60, 80), 2)

            # Title bar
            cv2.rectangle(combined, (0, 0), (total_width, 25), (30, 30, 45), -1)
            title = f"FRA Demo  |  Episode {ep+1}/{n_episodes}  |  Step {t}  |  Seed {seed + ep}"
            cv2.putText(combined, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 220), 1, cv2.LINE_AA)

            # Write frame (BGR for OpenCV)
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            total_frames += 1

            # Take action (simple policy for demo)
            if ttc < 1.5:
                action = 2  # BRAKE
            elif ttc < 3.0:
                action = 3  # LANE_CHANGE
            elif obs[2] < 25:
                action = 1  # ACCELERATE
            else:
                action = 0  # MAINTAIN

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated:
                episode_collision = info.get("collision", True)
                # Hold on collision frame for 1 second
                for _ in range(fps):
                    # Red flash
                    flash = combined.copy()
                    flash[:, :, 0] = np.minimum(flash[:, :, 0].astype(int) + 40, 255).astype(np.uint8)
                    writer.write(cv2.cvtColor(flash, cv2.COLOR_RGB2BGR))
                    total_frames += 1
                break

            if truncated:
                break

        total_episodes_done += 1
        if episode_collision:
            total_collisions += 1

        print(f"    Steps: {t+1} | Reward: {episode_reward:.1f} | "
              f"Collision: {episode_collision} | CR: {total_collisions/total_episodes_done:.3f}")

    writer.release()
    env.close()

    duration = total_frames / fps
    print(f"\n=== Demo Complete ===")
    print(f"Output: {output_path}")
    print(f"Frames: {total_frames} | Duration: {duration:.1f}s")
    print(f"Episodes: {n_episodes} | CR: {total_collisions/max(total_episodes_done,1):.3f}")


def main():
    parser = argparse.ArgumentParser(description="FRA Demo Video Capture")
    parser.add_argument("--output", default="demo.mp4", help="Output video path")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to record")
    parser.add_argument("--fps", type=int, default=5, help="Video FPS (lower = slower)")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--slow", action="store_true", help="Extra slow (2 FPS)")
    parser.add_argument("--g-max", type=float, default=10.0, help="A1 gradient norm bound")
    parser.add_argument("--bound", type=float, default=1.0, help="Prop 1 WDN bound")
    args = parser.parse_args()

    fps = 2 if args.slow else args.fps

    run_demo_capture(
        output_path=args.output,
        n_episodes=args.episodes,
        fps=fps,
        seed=args.seed,
        g_max=args.g_max,
        prop1_bound=args.bound,
    )


if __name__ == "__main__":
    main()
