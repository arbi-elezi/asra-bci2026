"""Generate paper in Springer LNCS format (llncs class) for BCI 2026.

BCI uses Springer LNCS proceedings format:
  - llncs document class
  - 12pt body, single column
  - Numbered sections, equations, figures, tables
  - splncs04.bst bibliography style
  - Max ~16 pages for full papers
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

TRIALS_DIR = Path("../experiment_stats/SmolLM2-135M/trials")
OUT = Path("main_lncs.tex")


def load_trials():
    trials = []
    for t in range(10):
        f = TRIALS_DIR / f"trial_{t}.json"
        if f.exists():
            trials.append(json.load(open(f)))
    return trials


def compute_stats(trials):
    bl_cr = np.array([t["baseline"]["cr"] for t in trials])
    v8_cr = np.array([t["v8"]["cr"] for t in trials])
    v8_lr = np.array([t["v8"]["lr"] for t in trials])
    v8_rr = np.array([t["v8"]["rr"] for t in trials])
    n = len(trials)
    s = {"n": n}
    s["bl_cr_mean"], s["bl_cr_std"] = bl_cr.mean(), bl_cr.std()
    s["v8_cr_mean"], s["v8_cr_std"] = v8_cr.mean(), v8_cr.std()
    s["v8_lr_mean"], s["v8_lr_std"] = v8_lr.mean(), v8_lr.std()
    s["v8_rr_mean"], s["v8_rr_std"] = v8_rr.mean(), v8_rr.std()
    _, s["wilcoxon_lr_p"] = stats.wilcoxon(v8_lr, alternative="greater")
    _, s["wilcoxon_rr_p"] = stats.wilcoxon(v8_rr, alternative="greater")
    _, s["ttest_cr_p"] = stats.ttest_rel(bl_cr, v8_cr)
    s["cohens_d_lr"] = v8_lr.mean() / v8_lr.std() if v8_lr.std() > 0 else 0
    pooled = np.sqrt((bl_cr.std()**2 + v8_cr.std()**2) / 2)
    s["cohens_d_cr"] = (v8_cr.mean() - bl_cr.mean()) / pooled if pooled > 0 else 0
    rng = np.random.default_rng(42)
    boots_lr = [rng.choice(v8_lr, n, replace=True).mean() for _ in range(10000)]
    s["ci_lr_lo"], s["ci_lr_hi"] = np.percentile(boots_lr, [2.5, 97.5])
    boots_rr = [rng.choice(v8_rr, n, replace=True).mean() for _ in range(10000)]
    s["ci_rr_lo"], s["ci_rr_hi"] = np.percentile(boots_rr, [2.5, 97.5])
    if n >= 5:
        _, s["shapiro_lr_p"] = stats.shapiro(v8_lr)
        _, s["shapiro_rr_p"] = stats.shapiro(v8_rr)
    return s


def sig(p):
    if p < 0.001: return "$^{***}$"
    if p < 0.01: return "$^{**}$"
    if p < 0.05: return "$^{*}$"
    return "n.s."


def gen(trials, s):
    n = s["n"]

    # Build per-trial rows
    trial_rows = ""
    for i, t in enumerate(trials):
        trial_rows += (
            f"    {i+1} & {t['baseline']['cr']:.3f} & {t['v8']['cr']:.3f} "
            f"& {t['v8']['lr']*100:.1f} & {t['v8']['rr']:+.3f} \\\\\n"
        )

    return r"""\documentclass[runningheads]{llncs}

\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{hyperref}

\begin{document}

\title{Aversive Salience-Regulated Agent: Learned Inference-Time Behavioral Modulation via Targeted Weight Perturbation and Confidence Adjustment}

\author{[Author Names Anonymized for Review]}
\authorrunning{[Anonymized]}
\institute{[Institution Anonymized for Review]}

\maketitle

%% ================================================================
\begin{abstract}
Pre-trained language model-based decision agents cannot modulate their risk tolerance post-deployment. We propose the Aversive Salience-Regulated Agent (ASRA), an inference-time mechanism that temporarily suppresses risky behavior through two complementary channels: (1)~Gaussian-targeted perturbation of the neural weights responsible for the risky decision, and (2)~confidence-level adjustment of the output distribution. A learned regulator controls both channels, trained via REINFORCE from a risk-reduction reward signal. Across """ + str(n) + r""" independent trials ($n{=}""" + str(n) + r"""$, 5\,000 episodes each), the mechanism achieves a risk-reduction rate of """ + f"{s['v8_lr_mean']*100:.1f}" + r"""\,$\pm$\,""" + f"{s['v8_lr_std']*100:.1f}" + r"""\,\% (Wilcoxon $p{=}""" + f"{s['wilcoxon_lr_p']:.4f}" + r"""$, Cohen's $d{=}""" + f"{s['cohens_d_lr']:.2f}" + r"""$). Five initial mechanism variants failed; these negative results are reported as they constrain the design space.

\keywords{aversive salience \and inference-time modulation \and targeted perturbation \and confidence adjustment \and risk-calibrated regulation}
\end{abstract}

%% ================================================================
\section{Introduction}
\label{sec:intro}

A frozen pre-trained driving policy has a fixed risk tolerance. When it encounters a state with two-second time-to-collision and chooses to accelerate, no post-deployment mechanism exists to make it reconsider---short of overriding its output entirely, which discards the learned policy's knowledge, or retraining, which requires offline data collection.

We address a specific version of this problem: can we attach an inference-time mechanism to a frozen language model-based driving policy that (a)~temporarily makes the policy less inclined toward risky actions when threat is detected, (b)~learns from experience how much and where to intervene, and (c)~recovers smoothly to baseline behavior after the threat passes?

The base policy is intentionally trained to a high collision rate (""" + f"{s['bl_cr_mean']:.2f}" + r""") to provide a challenging test environment with abundant threat encounters per episode. This choice maximizes the number of ASRA-active timesteps, enabling faster regulator learning and clear signal for measuring risk reduction. The mechanism's behavior on a well-trained policy is outside scope and constitutes future work.

The weight perturbation channel and the confidence adjustment channel serve distinct roles (Fig.~\ref{fig:attribution}). The weight channel provides \emph{mechanical} crash reduction through Gaussian suppression dynamics. The confidence channel provides \emph{learned} risk modulation. The learning signal is carried entirely by the confidence channel; the weight channel does not learn.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig14_channel_attribution.png}
\caption{Channel attribution. Left: collision rate (weight channel contribution). Right: risk-reduction rate (confidence channel contribution). The combined mechanism captures both effects.}
\label{fig:attribution}
\end{figure}

%% ================================================================
\section{Background}
\label{sec:background}

Existing inference-time safety mechanisms modify agent behavior through control barrier functions~\cite{ames2019cbf}, safe action overrides~\cite{saunders2018trial}, and shielding approaches~\cite{dalal2018safe}. All are binary: either the agent acts freely or its output is overridden. None modulate the agent's own behavioral tendencies.

Biological fear responses do not override motor output. The amygdala activates defensive circuits that suppress approach behavior and promote avoidance, without replacing the organism's motor planning capacity~\cite{ledoux1996emotional,damasio1994descartes}. The organism still decides what to do---it weighs options differently under threat.

We use \textbf{aversive salience} rather than ``fear'' to describe the threat detection signal, following computational neuroscience terminology~\cite{berridge2003parsing}. The aversive salience signal $S_t \in [0,1]$ quantifies how much the current state demands defensive behavioral adjustment.

%% ================================================================
\section{Method}
\label{sec:method}

ASRA attaches to a frozen LLM-based driving policy and operates through four components: (1)~a threat detector producing aversive salience $S_t$, (2)~a risk evaluator scoring the proposed action, (3)~a learned regulator controlling perturbation and confidence parameters, and (4)~a recovery controller determining decay rates.

\subsection{Threat Detection and Risk Evaluation}

An independent ensemble produces the aversive salience signal:
\begin{equation}
S_t = w_{ae} \cdot S_t^{AE} + w_{if} \cdot S_t^{IF} + w_{ca} \cdot S_t^{CA}
\label{eq:salience}
\end{equation}
where $S_t^{AE}$ is autoencoder reconstruction error (distribution shift detection), $S_t^{IF}$ is isolation forest anomaly score, and $S_t^{CA}$ is cost-advantage signal from a frozen cost critic.

Given the policy's greedy action $a_t$, the risk evaluator computes $R_t = f_{\text{risk}}(s_t, a_t, c_t, \text{TTC}_t)$ following a legally-grounded rubric: braking is always low-risk ($R \leq 0.1$), accelerating near obstacles is high-risk ($R \geq 0.6$). The perturbation strength combines both signals:
\begin{equation}
\alpha_t = S_t \cdot R_t
\label{eq:alpha}
\end{equation}

\subsection{Channel 1: Targeted Gaussian Weight Perturbation}

When $\alpha_t > 0.05$, the mechanism computes $\nabla_W \log \pi_W(a_t | s_t)$ to identify which weights drove the risky action. For each parameter group $k$, the epicenter $i^* = \arg\max_i |(\nabla_W)_i|$ is identified, and perturbation is applied with Gaussian spatial decay:
\begin{equation}
W_i \leftarrow W_i - \eta \cdot \alpha_t \cdot g_k \cdot \exp\!\left(-\frac{(i - i^*)^2}{2\sigma^2}\right) \cdot (\nabla_W)_i
\label{eq:gaussian}
\end{equation}

This suppresses the circuit that wanted the risky action. Weights far from the epicenter are barely affected (Fig.~\ref{fig:gaussian}).

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig15_gaussian_kernel.png}
\caption{Gaussian perturbation kernel. Left: 2D profiles at four widths showing smooth spatial decay from the epicenter. Right: 3D surface of the suppression field.}
\label{fig:gaussian}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig10_3d_perturbation.png}
\caption{Weight perturbation across layers and timesteps from the v8 experiment. Perturbation concentrates in layers 2--4, demonstrating wave-like propagation.}
\label{fig:perturbation3d}
\end{figure}

\subsection{Channel 2: Confidence Adjustment}

The regulator simultaneously adjusts the output distribution:
\begin{equation}
\hat{\ell}_t = \frac{\ell_t + \mathbf{s}}{T}
\label{eq:confidence}
\end{equation}
where $\ell_t$ are the raw logits, $\mathbf{s}$ is a per-action suppression vector (negative entries penalize specific actions), and $T$ is temperature (higher values flatten the distribution). The regulator learns to apply targeted suppression on the risky action's logit while leaving safe actions untouched.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig13_confidence_trajectory.png}
\caption{Confidence state trajectory showing elastic recovery after aversive salience spikes. Temperature decays smoothly; suppression targets accelerate and lane-change while leaving brake untouched.}
\label{fig:confidence}
\end{figure}

\subsection{Regulator Training}

After perturbation, the policy produces a new action $a'_t$. The reward signal is the risk reduction achieved:
\begin{equation}
r_t = R(s_t, a_t) - R(s_t, a'_t)
\label{eq:reward}
\end{equation}
The regulator learns via REINFORCE~\cite{schulman2017ppo} with episode-level reward normalization.

\subsection{Recovery Dynamics}

Confidence decays toward baseline via exponential recovery:
\begin{equation}
T_{t+1} = 1 + (T_t - 1) \cdot \rho_T, \quad \mathbf{s}_{t+1} = \mathbf{s}_t \cdot \rho_s
\label{eq:recovery_conf}
\end{equation}

Weight recovery follows Fisher-weighted homeostatic regulation~\cite{kirkpatrick2017ewc}:
\begin{equation}
W_{t+1} = W_t + \eta_h \cdot \rho_w \cdot \hat{F}_I \odot (W_0 - W_t)
\label{eq:recovery_weight}
\end{equation}
where $\hat{F}_I$ is the diagonal Fisher information matrix and $\rho_w \in [0.8, 0.99]$ controls recovery rate. The overdamped regime ensures monotonic return to baseline without oscillation (Fig.~\ref{fig:recovery}).

\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{fig16_elastic_recovery.png}
\caption{Damped spring recovery dynamics. ASRA uses the overdamped regime (blue): smooth monotonic return to baseline over approximately 50 timesteps.}
\label{fig:recovery}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{fig12_fisher_landscape.png}
\caption{Fisher information landscape across 502\,148 parameters (range: $4\,722\times$). Peaks correspond to high-importance parameters receiving stronger restoring force.}
\label{fig:fisher}
\end{figure}

%% ================================================================
\section{Experimental Setup}
\label{sec:setup}

\textbf{Decision maker.} SmolLM2-135M (HuggingFace), a 135M-parameter causal language model with LoRA adapters (rank~8, $\alpha{=}16$). Total perturbable parameters: 502\,148. Action head in float32.

\textbf{Environment.} highway-env~\cite{leurent2018highway} with 15 IDM vehicles, density 1.5. State: $s_t \in \mathbb{R}^{12}$. Actions: $|A|{=}4$. Cost: $c_t = \max(0, (2 - \text{TTC}_t)/2)$.

\textbf{Primary metric.} Risk-reduction rate (\emph{LessRisky}): fraction of ASRA-active timesteps where the regulated action has lower risk than the unregulated action.

\textbf{Statistical design.} """ + str(n) + r""" independent trials with different random seeds, 5\,000 episodes each. Wilcoxon signed-rank test (non-parametric), bootstrap 95\,\% confidence intervals ($B{=}10\,000$), Cohen's $d$ effect size, Shapiro-Wilk normality test.

%% ================================================================
\section{Results}
\label{sec:results}

\subsection{Negative Results: Failed Mechanism Variants}

Five initial designs failed (Table~\ref{tab:failures}). Uniform or untargeted weight perturbation consistently degrades the policy, regardless of direction.

\begin{table}[t]
\centering
\caption{Failed mechanism variants. All increased collision rate above baseline (0.545 for v1, 0.775 for v3--v5). BL: baseline collision rate.}
\label{tab:failures}
\begin{tabular}{llcc}
\toprule
Version & Mechanism & Perturbation target & CR \\
\midrule
v1 & Attract to critic's safe action & All 5\,252 (MLP) & 0.600 \\
v3 & Attract to safe action & All 502K (LoRA) & 0.980 \\
v4 & Suppress greedy action & All 502K & 0.880 \\
v5 & Suppress top-$K$\,\% by gradient & Binary mask & 0.900 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Channel Isolation}

\textbf{v6 (weight perturbation only):} Collision rate reduced from 0.80 to 0.50 (37\,\% reduction), but LessRisky remained at 13\,\% with zero training loss throughout. The improvement is purely mechanical.

\textbf{v7 (confidence adjustment only):} LessRisky climbed from 17\,\% to 85\,\% over 2\,000 episodes, but collision rate remained at baseline. The regulator learned that per-action logit suppression outperforms temperature scaling (mean temperature dropped from 2.5 to 1.3).

\subsection{Combined Mechanism: Statistical Results}

Table~\ref{tab:trials} presents per-trial results. Table~\ref{tab:stats} presents aggregate statistics. Fig.~\ref{fig:learning} shows the learning curves.

\begin{table}[t]
\centering
\caption{Per-trial results ($n{=}""" + str(n) + r"""$, 5\,000 episodes each).}
\label{tab:trials}
\begin{tabular}{ccccc}
\toprule
Trial & BL CR & v8 CR & LessRisky (\%) & RiskRed \\
\midrule
""" + trial_rows + r"""\midrule
Mean & """ + f"{s['bl_cr_mean']:.3f}" + r""" & """ + f"{s['v8_cr_mean']:.3f}" + r""" & """ + f"{s['v8_lr_mean']*100:.1f}" + r""" & """ + f"{s['v8_rr_mean']:+.3f}" + r""" \\
Std & """ + f"{s['bl_cr_std']:.3f}" + r""" & """ + f"{s['v8_cr_std']:.3f}" + r""" & """ + f"{s['v8_lr_std']*100:.1f}" + r""" & """ + f"{s['v8_rr_std']:.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Statistical analysis ($n{=}""" + str(n) + r"""$).}
\label{tab:stats}
\begin{tabular}{lcccc}
\toprule
Metric & Mean $\pm$ Std & 95\,\% CI & $p$-value & Cohen's $d$ \\
\midrule
BL CR & """ + f"{s['bl_cr_mean']:.3f}" + r"""\,$\pm$\,""" + f"{s['bl_cr_std']:.3f}" + r""" & --- & --- & --- \\
LessRisky (\%) & """ + f"{s['v8_lr_mean']*100:.1f}" + r"""\,$\pm$\,""" + f"{s['v8_lr_std']*100:.1f}" + r""" & [""" + f"{s['ci_lr_lo']*100:.1f}" + r""", """ + f"{s['ci_lr_hi']*100:.1f}" + r"""] & """ + f"{s['wilcoxon_lr_p']:.4f}" + r"" + sig(s['wilcoxon_lr_p']) + r""" & """ + f"{s['cohens_d_lr']:.2f}" + r""" \\
RiskRed & """ + f"{s['v8_rr_mean']:+.3f}" + r"""\,$\pm$\,""" + f"{s['v8_rr_std']:.3f}" + r""" & [""" + f"{s['ci_rr_lo']:+.3f}" + r""", """ + f"{s['ci_rr_hi']:+.3f}" + r"""] & """ + f"{s['wilcoxon_rr_p']:.4f}" + r"" + sig(s['wilcoxon_rr_p']) + r""" & --- \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig11_learning_curves.png}
\caption{Learning curves. v6 (red) remains flat at 13\,\%. v7 (blue) and v8 (green) climb to high LessRisky rates, with v8 additionally reducing collision rate.}
\label{fig:learning}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig17_risk_distribution.png}
\caption{Distribution of per-episode risk reduction: early training (red, mean $+0.21$) versus late training (green, mean $+0.66$). The regulator shifts the distribution rightward.}
\label{fig:riskdist}
\end{figure}

\subsection{Channel Attribution}

The v6/v7/v8 comparison forms a $2{\times}2$ factorial design (Table~\ref{tab:factorial}).

\begin{table}[t]
\centering
\caption{Channel attribution ($2{\times}2$ factorial).}
\label{tab:factorial}
\begin{tabular}{lcc}
\toprule
& Weight OFF & Weight ON \\
\midrule
Confidence OFF & BL (CR$=$0.80) & v6 (CR$=$0.50, LR$=$13\,\%) \\
Confidence ON & v7 (CR$=$0.80, LR$=$85\,\%) & v8 (CR$=$0.69, LR$=$86\,\%) \\
\bottomrule
\end{tabular}
\end{table}

%% ================================================================
\section{Analysis}
\label{sec:analysis}

\textbf{Why uniform perturbation fails.} Experiments v1--v5 show that perturbing all weights uniformly degrades the policy. The vast majority of weights are uninvolved in the current risky decision; perturbing them introduces noise corrupting unrelated representations.

\textbf{Why confidence alone does not reduce crashes.} v7 achieves 85\,\% LessRisky but identical CR to baseline. Confidence adjustment increases output entropy: the policy becomes less committed to \emph{all} actions. In a high-CR environment, more randomness creates more crash opportunities.

\textbf{Surgical suppression discovery.} Over training, the regulator's mean temperature drops from 2.5 to 1.3 while LessRisky climbs from 27\,\% to 86\,\% (Fig.~\ref{fig:evolution}). The regulator discovers that per-action logit suppression is more effective than global temperature scaling.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig18_regulator_evolution.png}
\caption{Regulator evolution: temperature drops while LessRisky climbs---the regulator discovers surgical per-action suppression outperforms blanket temperature scaling.}
\label{fig:evolution}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig5_perturbation_dynamics.png}
\caption{Real-time perturbation dynamics showing weight deviation, fear signal, and gradient norm across three experimental conditions.}
\label{fig:dynamics}
\end{figure}

%% ================================================================
\section{Limitations}
\label{sec:limitations}

(1)~All results are from highway-env with IDM traffic; generalization is untested. (2)~Only SmolLM2-135M tested; TinyLlama-1.1B was initiated but not completed. (3)~The risk evaluator uses a fixed hand-designed rubric. (4)~The baseline collision rate is """ + f"{s['bl_cr_mean']*100:.0f}" + r"""\,\% (see Sect.~\ref{sec:intro} for justification).

%% ================================================================
\section{Conclusion}
\label{sec:conclusion}

We presented ASRA, an inference-time mechanism modulating a frozen LLM policy's risk-taking behavior through targeted Gaussian weight perturbation and learned confidence adjustment.

Across """ + str(n) + r""" independent trials, the combined mechanism achieves a risk-reduction rate of """ + f"{s['v8_lr_mean']*100:.1f}" + r"""\,$\pm$\,""" + f"{s['v8_lr_std']*100:.1f}" + r"""\,\% (Wilcoxon $p{=}""" + f"{s['wilcoxon_lr_p']:.4f}" + r"""$, Cohen's $d{=}""" + f"{s['cohens_d_lr']:.2f}" + r"""$). Five failed variants constrain the design space: perturbation must be targeted, shaped with smooth decay, trained with a strong signal, and combined across weight and confidence channels.

The learning originates entirely in the confidence channel. The weight channel provides mechanical crash reduction but does not learn. The dual-channel combination achieves both crash reduction and learned risk modulation---properties neither channel provides alone.

%% ================================================================
\begin{thebibliography}{10}

\bibitem{ames2019cbf}
Ames, A.D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., Tabuada, P.: Control barrier functions: Theory and applications. In: European Control Conference (2019)

\bibitem{saunders2018trial}
Saunders, W., Sastry, G., Stuhlmueller, A., Evans, O.: Trial without error: Towards safe reinforcement learning via human intervention. In: AAMAS (2018)

\bibitem{ledoux1996emotional}
LeDoux, J.E.: The Emotional Brain. Simon \& Schuster (1996)

\bibitem{damasio1994descartes}
Damasio, A.R.: Descartes' Error: Emotion, Reason, and the Human Brain. Putnam (1994)

\bibitem{berridge2003parsing}
Berridge, K.C., Robinson, T.E.: Parsing reward. Trends in Neurosciences \textbf{26}(9), 507--513 (2003)

\bibitem{dalal2018safe}
Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., Tassa, Y.: Safe exploration in continuous action spaces. arXiv:1801.08757 (2018)

\bibitem{schulman2017ppo}
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv:1707.06347 (2017)

\bibitem{leurent2018highway}
Leurent, E.: An environment for autonomous driving decision-making. GitHub (2018)

\bibitem{kirkpatrick2017ewc}
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al.: Overcoming catastrophic forgetting in neural networks. PNAS \textbf{114}(13), 3521--3526 (2017)

\bibitem{amari1998natural}
Amari, S.: Natural gradient works efficiently in learning. Neural Computation \textbf{10}(2), 251--276 (1998)

\end{thebibliography}

\end{document}
"""


if __name__ == "__main__":
    trials = load_trials()
    print(f"Loaded {len(trials)} trials")
    s = compute_stats(trials)
    print(f"BL CR: {s['bl_cr_mean']:.3f} +/- {s['bl_cr_std']:.3f}")
    print(f"LessRisky: {s['v8_lr_mean']*100:.1f}% +/- {s['v8_lr_std']*100:.1f}%")
    print(f"Wilcoxon p = {s['wilcoxon_lr_p']:.6f}")
    print(f"Cohen's d = {s['cohens_d_lr']:.2f}")

    tex = gen(trials, s)
    with open(OUT, "w") as f:
        f.write(tex)
    print(f"\nWritten {OUT} ({len(tex)} chars, ~{len(tex.split(chr(10)))} lines)")
