# Task Spec — Survey: "Diffusion Models in Finance"

## Objective
Produce a comprehensive, citation-grounded survey/review paper on **diffusion models
(score-based / DDPM / SDE generative models) applied to finance**, then condense it
into an ICAIF-ready ~9-page ACM `sigconf` submission.

Strategy (user-chosen): **build big, then condense.** Coverage is broad — the whole
area, not a single sub-application.

## Target venue
- Primary: **ICAIF** (ACM International Conference on AI in Finance) — full paper,
  ACM `sigconf`, ~8–9 pages + references.
- Secondary artifact: extended comprehensive survey (30+ pp) for arXiv / journal.

## Scope (cover everything big)
1. Foundations: DDPM, score matching / score-based SDEs, conditional & guided
   diffusion, latent diffusion, consistency/flow-matching variants.
2. Synthetic financial data & time-series generation (returns, prices, tabular,
   limit-order-book, multivariate, high-frequency).
3. Risk: scenario generation, stress testing, tail/VaR/ES, regime simulation.
4. Pricing & hedging: derivative pricing, calibration, deep hedging, solving
   financial PDEs/SDEs with score/diffusion methods.
5. Portfolio & trading: portfolio optimization, market simulation, forecasting,
   anomaly/fraud detection, RL-adjacent uses.
6. Evaluation: fidelity/diversity metrics, stylized facts, backtesting, privacy.
7. Open problems: data scarcity, non-stationarity, evaluation rigor, regulation,
   compute, theoretical guarantees.

## Milestones
- M1: Literature corpus assembled & verified (≥80 real, verified references).
- M2: Taxonomy + annotated outline locked.
- M3: Full-length section drafts (foundations → applications → evaluation → open problems).
- M4: Complete comprehensive draft + verified .bib.
- M5: Condensed ICAIF ~9pp ACM paper + camera-ready checks.

## Success criteria
- Every citation corresponds to a REAL, verifiable paper (title+venue+year checked).
- Clear taxonomy; each application area covered with concrete methods + results.
- Honest discussion of limitations and open problems.
- ICAIF version fits page limit in ACM `sigconf` format and reads as a coherent survey.

## Hard rules (from Deli_AutoResearch framework)
- Persist ALL progress to state/ files; each iteration curates state, never relies on chat memory.
- Citation verification every ≤20 entries, never batched at the end.
- Max 5 large files touched per iteration; no file > 300 lines.
- Add direction diversity over digging one deeper when stalled.
- Decisions logged to logs/work.jsonl with level=decision.
