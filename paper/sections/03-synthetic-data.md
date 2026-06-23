# 3. Synthetic Financial Data and Time-Series Generation

The largest cluster of diffusion-in-finance work concerns generating synthetic
data that passes for real. We organize it by data type and by the fidelity
target—reproducing *stylized facts*, supporting *downstream tasks*, or
preserving *privacy*.

## 3.1 Univariate returns and prices

The core challenge is matching stylized facts—heavy tails, volatility
clustering, the leverage effect—that simpler generators miss. Two design
philosophies have emerged. The first reshapes the data so that off-the-shelf
diffusion applies: \citet{takahashi2024synthfts} apply a wavelet transform to map
multivariate series (price, volume, spread) into image-like representations,
train a DDPM, and invert the transform, recovering fat tails and volatility
clustering. The second bakes financial structure into the *forward* process:
\citet{kim2025gbmdiff} replace the standard Gaussian corruption with a geometric
Brownian motion–informed noising process, improving heavy-tail and
volatility-clustering fidelity relative to a vanilla DDPM. The recurring lesson
is that *inductive bias matters*: financial priors in the noise process or
representation outperform architecture tuning alone.

## 3.2 High-dimensional and cross-sectional generation

Portfolio and risk applications need *joint* distributions over many assets with
realistic dependence. \citet{chen2025difffactor} embed a latent factor structure
into a diffusion model, generating high-dimensional returns that respect a
factor covariance while retaining diffusion's flexibility; they also provide
theoretical guarantees and show gains in small-sample regimes, where the factor
prior regularizes estimation. This factor-structured view connects directly to
scenario generation (§5) and portfolio construction (§7).

## 3.3 Controllable and conditional generation

Conditioning turns a generator into a scenario engine. CoFinDiff
\cite{tanaka2025cofindiff} conditions on target trend and realized volatility,
producing series that match specified controls with lower error and more
diversity than GAN/VAE baselines, and demonstrates downstream gains in hedging.
Controllability is the property that most distinguishes diffusion from earlier
generators and recurs in market simulation (§4) and stress testing (§5).

## 3.4 Tabular financial data: credit, fraud, and mixed types

Much financial data is tabular and mixed-type (numeric + categorical). TabDDPM
\cite{kotelnikov2023tabddpm} established mixed-type tabular diffusion via
combined Gaussian and multinomial processes. FinDiff \cite{sattarov2023findiff},
presented at ICAIF, specializes this to financial tables—consumer credit,
public-sector expenditure, fund composition—reporting strong fidelity, utility,
and *privacy* relative to TVAE, CTGAN, and TabDDPM. This line is the natural home
for fraud and credit-risk augmentation, where the minority (fraud) class is
scarce and privacy constraints are binding; it also connects to the anomaly-
detection literature surveyed by \citet{liu2025anomalysurvey}.

## 3.5 Matching real-world quality and downstream utility

Fidelity is ultimately judged by use. Fiaingen \cite{rozanec2025fiaingen}
targets synthetic series that match real-world data quality while keeping
generation fast, reporting train-on-synthetic / test-on-real performance close to
training on real data—the operational bar for synthetic data. Comparative studies
of deep generative models for portfolio and risk modeling
\cite{hounwanou2025deepgen} position diffusion against TimeGAN and VAEs as a
privacy-preserving substitute, clarifying where each family wins.

## 3.6 Diffusion versus GAN/VAE baselines

Throughout this area, the relevant non-diffusion baselines are Quant GANs
\cite{wiese2020quantgan}, Sig-Wasserstein GANs \cite{ni2021sigwgan}, and
structured-noise/signature-MMD models \cite{lu2024structnoise}. The reported
pattern is consistent: diffusion models train more stably and condition more
naturally, with competitive or better stylized-fact reproduction, at the cost of
slower sampling (§2.4, §9). We caution that cross-paper comparison is hampered by
inconsistent datasets and metrics—a problem we return to in §8.

## 3.7 The tabular-diffusion toolkit and imbalanced finance

The financial-tabular line sits atop a fast-moving general toolkit. Beyond
TabDDPM, latent-space score-based synthesis (TabSyn \cite{zhang2024tabsyn}), its
rectified-flow counterpart RecTable \cite{fuchi2025rectable}, and broader
comparative surveys \cite{challagundla2025tabsurvey} now define the state
of the art that financial models specialize. Two finance-specific extensions of
FinDiff target the field's defining difficulty—severe class imbalance: Imb-FinDiff
\cite{sattarov2024imbfindiff} adds a joint denoising-plus-label objective to
synthesize minority (e.g., fraud) classes, and EntTabDiff
\cite{liu2024enttabdiff} conditions generation on *entities* (companies,
individuals) to improve fidelity and reduce entity-level privacy leakage. For
fraud specifically, EmDT \cite{kuo2026emdt} trains per-cluster diffusion
transformers, and latent flow-based diffusion \cite{ihsan2025latentflow} targets
minority oversampling across finance and other domains. Actuarial/insurance use
is nascent but methodologically adjacent—e.g., diffusion for survival
(time-to-event) data \cite{brockschmidt2025survdiff} maps onto mortality and
claims modeling.

The non-diffusion frontier also keeps advancing—e.g., macro-conditioned GAN
generators with adversarial perturbations for robust trading
\cite{xia2026bayesrobust}—and remains the comparison point against which
diffusion's stability and conditioning advantages are judged.
