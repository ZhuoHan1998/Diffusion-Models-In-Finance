# 7. Forecasting and Portfolio Construction

Diffusion models are *probabilistic* forecasters: they produce predictive
distributions, not point forecasts, which is precisely what portfolio
construction and risk budgeting consume. This section covers multivariate
forecasting and its use in allocation.

## 7.1 Multivariate probabilistic forecasting

TimeGrad \cite{rasul2021timegrad} established autoregressive diffusion for
multivariate probabilistic time-series forecasting and remains the standard
baseline. In finance, Diffolio \cite{cho2025diffolio} specializes the idea: a
denoising network with hierarchical (asset-level and market-level) attention,
plus a correlation-guided regularizer informed by a stable estimate of the target
correlation matrix, so that the predictive distribution respects cross-sectional
dependence. On industry-portfolio returns it improves both forecasting accuracy
and downstream portfolio performance over probabilistic baselines—evidence that
modeling dependence explicitly, rather than per-asset, is what carries the gain.

## 7.2 From forecasts to portfolios

Two routes turn a generative model into an allocation. The *predict-then-optimize*
route feeds the predictive distribution into a mean–variance or growth-optimal
objective; Diffolio \cite{cho2025diffolio} reports higher Sharpe ratios for the
tangency portfolio and higher certainty equivalents for the growth-optimal
portfolio constructed from its forecasts. The *direct* route learns the policy
end-to-end: \citet{aghapour2025dynport} solve dynamic portfolio selection with
score-based diffusion in a model-free way, outperforming classical Markowitz
allocation, while \citet{choudhary2025diffaugrl} (§5.2) optimize policies against
diffusion-generated stress. These build on the general paradigm of *decision-making
as conditional generation*—Diffuser \cite{janner2022diffuser} and Decision
Diffuser \cite{ajay2022decisiondiffuser}, surveyed for RL by
\citet{zhu2023diff4rl}—imported into trading and allocation, where multimodal
action distributions and long-horizon planning are natural fits (§9.5). The same
idea now appears in its flow-matching form: FlowHFT \cite{li2025flowhft} learns a
flow-matching imitation-learning *policy* for high-frequency trading that adapts
across diverse market regimes, illustrating that the decision-as-generation
program extends naturally from diffusion to the interpolant/flow family (§2.4b).
Synthetic-path augmentation is a third, indirect
route: structured-noise signature models have been used to train RL portfolio
agents \cite{lu2024structnoise}, and broad comparisons position diffusion/VAE/GAN
synthetic returns as privacy-preserving training data for allocation and risk
\cite{hounwanou2025deepgen}.

## 7.3 Forecasting in alternative representations

Forecasting need not operate on raw returns. \citet{lee2025charts} treat price
history as chart *images* and use a text-to-image diffusion model to forecast
future chart patterns from current charts and instruction prompts—an unusual but
illustrative case of importing a vision-diffusion pipeline wholesale, and a
reminder that representation choice (§3.1, §4.2) is a recurring design lever.

## 7.4 Summary

For decision-making, diffusion's probabilistic and dependence-aware forecasts are
its main asset. The strongest results couple a financial inductive bias
(hierarchical attention, correlation guidance, factor structure) to the
generator; the open questions are non-stationary generalization and honest
out-of-sample evaluation (§8–§9), since portfolio metrics are especially
vulnerable to look-ahead bias and overfitting to a backtest.

## 7.5 Relational forecasting, factor augmentation, and factor-conditioned allocation

Equity forecasting has spawned a cluster of diffusion models. DiffSTOCK
\cite{daiya2024diffstock} conditions a DDPM on indicators and inter-stock
relations for relational market prediction, while DiffsFormer
\cite{gao2024diffsformer} uses a diffusion transformer to *augment* scarce stock
factors. For allocation, factor-conditioned diffusion learns the cross-sectional
return distribution conditioned on firm characteristics and optimizes
mean–variance/mean-CVaR portfolios \cite{gao2025factorcond}, while
\citet{bagchi2026factordim} analyze the bias–variance tradeoff in the number of
conditioning factors—an explicitly econometric lens on diffusion portfolios.
General style-guided TS diffusion \cite{sun2025dsdiffusion} supplies further
backbone machinery these models draw on.
