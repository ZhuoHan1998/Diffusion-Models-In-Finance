# 11. Summary Table: Method × Task × Data × Evaluation

The following table consolidates the finance-specific diffusion literature surveyed
above. "Cond." marks models with explicit conditioning/controllability. "Code"
reflects an author-released implementation where stated in the source.

| Work | Task | Data / market | Method family | Cond. | Reported evaluation |
|------|------|---------------|---------------|:----:|---------------------|
| Takahashi & Mizuno \cite{takahashi2024synthfts} | Synthetic gen | Equity prices/vol/spread | DDPM + wavelet→image | – | Stylized facts (tails, clustering) |
| Kim et al. \cite{kim2025gbmdiff} | Synthetic gen | Equity returns | GBM-informed forward SDE | – | Heavy tails, vol clustering |
| Chen et al. \cite{chen2025difffactor} | Synthetic gen (hi-dim) | Cross-section of returns | Factor-structured score | – | Theory + small-sample utility |
| CoFinDiff \cite{tanaka2025cofindiff} | Controllable gen | Financial TS | Conditional diffusion | ✓ | Control error, hedging downstream |
| Fiaingen \cite{rozanec2025fiaingen} | Synthetic gen | Financial TS | Diffusion | – | TSTR ≈ real, speed |
| FinDiff \cite{sattarov2023findiff} | Tabular gen | Credit/expenditure/funds | Mixed-type diffusion | ✓ | Fidelity, utility, privacy |
| DP-FinDiff \cite{sattarov2025dpfindiff} | Private tabular gen | Financial/medical tables | DP diffusion | ✓ | Utility @ fixed DP budget |
| DP-FedTabDiff \cite{sattarov2024dpfedtabdiff} | Private tabular gen | Tabular | DP + federated diffusion | ✓ | Fidelity under DP |
| TRADES \cite{berti2025trades} | LOB simulation | Order flow (equities) | Transformer diffusion | ✓ | Predictive score |
| Painting the Market \cite{backhouse2025painting} | LOB sim/forecast | LOB (equities) | Image + inpainting diffusion | ✓ | LOB-Bench |
| DiffVolume \cite{wang2025diffvolume} | LOB volume gen | LOB volume | Conditional diffusion | ✓ | Distributional + counterfactual |
| DigMA \cite{huang2024digma} | Market generation | Order flow | Cond. diffusion + meta-agent | ✓ | Realism, HFT utility |
| Beyond Monte Carlo \cite{lesniewski2024beyondmc} | Market dynamics sim | Multi-asset | Diffusion | – | Covariance realism |
| OT power law \cite{hashimoto2025otpowerlaw} | Market sim eval | Artificial market | OT diagnostic | – | Power-law attribution |
| Choudhary et al. \cite{choudhary2025diffaugrl} | Stress + portfolio | Multi-asset | Diffusion stress + RL | ✓ | Robust portfolio metrics |
| Zhao & Shen \cite{zhao2025exotic} | Pricing (exotic) | Option underlyings | Conditional diffusion | ✓ | Pricing error (Euro/Asian) |
| Jin & Agarwal \cite{jin2025ivsurface} | IV surface | Options | Cond. DDPM + arb penalty | ✓ | Arb-free, MAE vs GAN |
| Gao et al. \cite{gao2025sdepaths} | SDE path gen | Generic SDEs | Conditional diffusion | ✓ | Path distribution fit |
| Diffolio \cite{cho2025diffolio} | Forecasting/portfolio | 12 industry portfolios | Hierarchical-attn diffusion | ✓ | Accuracy, Sharpe, CE |
| Aghapour et al. \cite{aghapour2025dynport} | Dynamic portfolio | Multi-asset | Score-based, model-free | – | vs Markowitz |
| Lee et al. \cite{lee2025charts} | Forecasting (charts) | Equity charts | Text-to-image diffusion | ✓ | Trend forecast |
| Hounwanou & Gaba \cite{hounwanou2025deepgen} | Synthetic for risk | Returns | GAN/VAE (baseline study) | – | Portfolio/risk utility, privacy |

**Cross-domain anchors used as backbones/baselines.** TimeGrad
\cite{rasul2021timegrad}, CSDI \cite{tashiro2021csdi}, SSSD
\cite{alcaraz2022sssd}, Diffusion-TS \cite{yuan2024diffusionts}, and TSDiff
\cite{kollovieh2023tsdiff} are the general time-series diffusion models most
often adapted to finance; TabDDPM \cite{kotelnikov2023tabddpm} for tables; DiT
\cite{peebles2022dit} for transformer backbones; Diffuser \cite{janner2022diffuser}
and Decision Diffuser \cite{ajay2022decisiondiffuser} for the decision/planning
route (cf. the RL survey \cite{zhu2023diff4rl}); and Quant GANs
\cite{wiese2020quantgan}, Sig-WGAN \cite{ni2021sigwgan}, and signature-MMD models
\cite{lu2024structnoise} as the non-diffusion generative baselines. Deep Hedging
\cite{buehler2019deephedging} is the canonical downstream decision problem these
generators feed.

**Reading the table.** Two patterns stand out. First, *conditioning is the norm*
in the most recent and most finance-specific work—controllability is the property
the field actually wants. Second, *evaluation is heterogeneous*: no two rows share
an identical protocol, which is precisely the comparability problem of §8.
