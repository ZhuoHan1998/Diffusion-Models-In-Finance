# Annotated Outline — "Diffusion Models in Finance: A Survey"

Status: DRAFT v0.1 (iter 1). Refs in [brackets] are findings.jsonl keys.
Comprehensive version first; ICAIF condensation derived later (M5).

## Proposed taxonomy (two axes)
- **Axis A — Methodological family:** DDPM (discrete) · score-based SDE (continuous) ·
  conditional/guided (classifier / classifier-free) · latent diffusion ·
  fast-sampling (DDIM/consistency/flow-matching) · structure-informed
  (GBM/factor/PDE-aware).
- **Axis B — Financial task:** (1) synthetic data & time-series generation ·
  (2) market microstructure / LOB simulation · (3) risk & scenario generation ·
  (4) pricing, hedging & calibration · (5) forecasting & portfolio construction ·
  (6) tabular / fraud / credit / privacy.

## 1. Introduction
- Why generative models in finance: data scarcity, privacy, non-stationarity,
  stress scenarios, simulation for backtesting.
- Why diffusion now: stable training vs GANs, mode coverage, flexible conditioning.
- Scope, contributions, taxonomy, reading guide.
- Contrast with prior GAN/VAE-centric surveys (gap this survey fills).

## 2. Background: diffusion & score-based generative models
- 2.1 Forward/reverse diffusion; DDPM objective [ho2020ddpm, sohldickstein2015]
- 2.2 Score matching & NCSN [song2019ncsn]
- 2.3 SDE unification; probability-flow ODE [song2021sde]
- 2.4 Sampling & acceleration: DDIM, consistency, flow matching
  [songddim2021, song2023consistency, lipman2023flow, karras2022edm]
- 2.5 Conditioning & guidance [dhariwal2021beatgan, ho2022cfg]; latent diffusion [rombach2022ldm]
- 2.6 Diffusion for sequences/tables (bridge to finance)
  [rasul2021timegrad, tashiro2021csdi, kotelnikov2023tabddpm]
- 2.7 Why finance is hard: stylized facts, non-stationarity, low SNR, regime shifts.

## 3. Synthetic financial data & time-series generation
- 3.1 Univariate returns/prices; stylized facts [takahashi2024synthfts, kim2025gbmdiff]
- 3.2 High-dimensional cross-section / factor structure [chen2025difffactor]
- 3.3 Tabular financial data, credit, fraud [sattarov2023findiff, cand_entitytab, kotelnikov2023tabddpm]
- 3.4 Quality vs real data; downstream-task utility [cand_fiaingen]
- TODO: QuantGAN/Fin-GAN baselines; conditional generation by regime.

## 4. Market microstructure & limit-order-book simulation
- 4.1 Order-flow generation [berti2025trades]
- 4.2 Image/inpainting LOB representations [backhouse2025painting]
- 4.3 Volume / depth generation [cand_diffvolume]
- 4.4 Counterfactual / responsive simulation; agent-based use [cand DiffLOB]
- Eval: LOB-Bench, market-impact realism, responsiveness.

## 5. Risk, scenario generation & stress testing
- 5.1 Conditional crash/tail scenario generation; VaR/ES/CVaR
- 5.2 Robust portfolio under stress (diffusion + RL) [cand_diffaugrl]
- 5.3 Hard-constraint / guided scenario generation
- 5.4 Compressed-sensing / inverse-problem framings in finance
- TODO: regulatory stress-testing context (CCAR/EBA); regime simulation.

## 6. Pricing, hedging & calibration
- 6.1 Risk-neutral generation & measure change (epsilon-shift) [cand risk-neutral pricing]
- 6.2 Exotic / path-dependent option valuation [cand_exoticopt]
- 6.3 Implied-volatility-surface modeling/forecasting [cand_ivsurface]
- 6.4 Deep hedging & neural-SDE pricing (bridge) — TODO Buehler; robust neural-SDE
- 6.5 Diffusion/score methods for financial PDE/HJB — TODO

## 7. Forecasting & portfolio construction
- 7.1 Multivariate probabilistic forecasting [cho2025diffolio, rasul2021timegrad]
- 7.2 Dynamic/model-free portfolio selection [aghapour2025dynport]
- 7.3 Correlation/dependence-aware generation; Sharpe/CE evaluation.

## 8. Evaluation, benchmarks & reproducibility
- 8.1 Stylized-fact checklists; distributional fidelity.
- 8.2 Discriminative/predictive scores; train-on-synthetic-test-on-real.
- 8.3 LOB-Bench and task-specific benchmarks.
- 8.4 Privacy/membership-inference; differential privacy. — TODO refs
- 8.5 Reproducibility, data availability, compute cost.

## 9. Open problems & research directions
- Theoretical guarantees under non-stationarity; sample complexity in low-data.
- Faithful tail/extreme-event modeling; rare-event conditioning.
- Causality & intervention vs pure generation; counterfactuals.
- Evaluation standardization; leakage & look-ahead bias.
- Regulation, auditability, model risk; privacy.
- Efficiency: few-step sampling for real-time/HFT.

## 10. Conclusion

## Appendices (comprehensive version only)
- A. Notation & diffusion math reference.
- B. Summary table: method × task × data × eval × code-availability.
- C. Datasets used across the literature.

---
### Coverage gaps to close in iter 2+ (drives directions_tried.untried_next)
surveys to position against · deep hedging / neural SDE · PDE solving ·
anomaly/fraud · evaluation metrics & privacy refs · GAN/VAE baselines ·
crypto/HFT/vol-surface · macro/credit/insurance/ESG.
