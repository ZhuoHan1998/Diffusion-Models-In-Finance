# 9. Open Problems and Research Directions

We close the technical discussion with the problems we believe most determine
whether diffusion models move from promising results to dependable financial
tools.

## 9.1 Tail and extreme-event fidelity

The decision-relevant mass in finance lives in the tails, yet several of the
strongest results degrade precisely there—exotic-option accuracy falls on extreme
scenarios \cite{zhao2025exotic}, and risk numbers inherit any tail thinning of
the generator (§5). Open questions: training objectives and noise schedules that
explicitly target tail accuracy; rare-event and importance-weighted conditioning;
and theoretical control of tail error. Structure-informed designs (GBM noising
\cite{kim2025gbmdiff}, factor priors \cite{chen2025difffactor}) are promising
levers but not yet a solution.

## 9.2 Non-stationarity and generalization guarantees

Financial distributions drift, and the scenarios we most need are the ones least
represented in history. Most evaluations are in-distribution (§8.5), so reported
performance likely overstates deployed performance. We need methods—and
benchmarks—for *out-of-regime* generalization, and theory bounding generative
error under distribution shift. Existing guarantees are narrow (e.g., the
small-sample factor-structure results of \citet{chen2025difffactor}).

## 9.3 Hard economic constraints

Current pricing work *penalizes* arbitrage \cite{jin2025ivsurface} or builds in
the change of measure approximately (§6). Enforcing *exact* no-arbitrage and the
martingale property under $\mathbb{Q}$ by construction—via constrained sampling,
projection, or guidance with hard constraints—remains open and is a prerequisite
for front-office adoption.

## 9.4 Efficiency for real-time use

High-frequency and large-scale risk applications cannot afford thousand-step
sampling. Few-step methods—DDIM \cite{songddim2021}, consistency models
\cite{song2023consistency}, flow matching \cite{lipman2023flow}—are the obvious
path, but their fidelity cost on heavy-tailed financial targets is largely
unmeasured. Real-time responsive market simulation (§4.4) is especially demanding.

## 9.5 Stochastic control and decision integration

The HJB/optimal-control view of diffusion \cite{berner2022optcontrol} and the
ability to learn unknown SDE dynamics \cite{gao2025sdepaths} hint at diffusion
samplers as solvers for high-dimensional investment, hedging, and execution
control problems. Tightly coupling generation with the downstream decision—as in
diffusion-augmented RL \cite{choudhary2025diffaugrl}—is a promising but immature
template.

## 9.6 Causality and counterfactuals

Most models are correlational generators. Counterfactual microstructure
generation \cite{wang2025diffvolume} and controllable market agents
\cite{huang2024digma} gesture toward intervention, but principled causal
diffusion—answering "what if this policy/shock occurred"—is largely unexplored
and is what stress testing ultimately requires.

## 9.7 Governance: model risk, auditability, regulation, privacy

Deploying generative models for capital, valuation, or surveillance raises
model-risk-management duties: validation, monitoring, explainability, and audit
trails. Synthetic-data privacy is currently evaluated with empirical proxies
\cite{sattarov2023findiff} rather than formal guarantees. Bridging the gap to
supervisory expectations (§5.4) is as much a research problem as an engineering
one, and we expect it to gate real-world use more than raw model quality does.

## 9.8 Standardized benchmarks

Underlying all of the above is the absence of shared, finance-specific
benchmarks with chronological, regime-aware splits. Community benchmarks did more
to advance vision diffusion than any single architecture; an analogous suite—
covering returns, LOB \cite{backhouse2025painting}, tabular \cite{sattarov2023findiff},
and risk tasks—would do the same here.
