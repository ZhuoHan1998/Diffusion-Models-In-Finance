# 1. Introduction

Generative modeling has become a central tool in quantitative finance. The
practical drivers are structural rather than fashionable: financial data are
*scarce* in the regimes that matter most (crises, regime transitions, rare
events), *sensitive* (client-level and proprietary records cannot be shared
freely), and *non-stationary* (the data-generating process drifts with policy,
liquidity, and market structure). Synthetic data that faithfully reproduces the
statistical signature of markets can relax all three constraints at once: it
augments thin datasets, substitutes for privacy-restricted records, and supplies
controllable scenarios for backtesting and stress testing
\cite{hounwanou2025deepgen, sattarov2023findiff}.

For roughly a decade this need was met primarily by generative adversarial
networks (GANs) and variational autoencoders (VAEs). Quant GANs
\cite{wiese2020quantgan} showed that temporal convolutional GANs could reproduce
volatility clustering and fat tails; signature-based variants such as
Sig-Wasserstein GANs \cite{ni2021sigwgan} and structured-noise/MMD approaches
\cite{lu2024structnoise} improved training stability and path realism. Yet GANs
remain difficult to train, prone to mode collapse, and awkward to condition,
while VAEs tend to oversmooth the heavy tails that dominate financial risk.

**Why diffusion, why now.** Denoising diffusion probabilistic models (DDPMs)
\cite{ho2020ddpm} and the closely related score-based models
\cite{song2019ncsn}, unified under a stochastic differential equation (SDE) view
\cite{song2021sde}, offer a different trade-off: stable likelihood-based
training, strong mode coverage, and—crucially for finance—native, flexible
*conditioning* through guidance \cite{dhariwal2021beatgan, ho2022cfg}. They also
sit naturally beside the mathematical objects quantitative finance already uses.
A diffusion model *is* an SDE with a learned drift; sampling is the numerical
solution of a reverse-time SDE or its probability-flow ODE; and the training
objective has a control-theoretic reading through a Hamilton–Jacobi–Bellman
equation on the log-density \cite{berner2022optcontrol}. For a field built on
Itô calculus, Monte Carlo, and stochastic control, this is a comfortable fit—and
it is the reason diffusion methods have, since roughly 2023, spread across nearly
every quantitative-finance task.

**Scope of this survey.** We review the use of diffusion and score-based
generative models across the breadth of finance, organized along two axes. The
*methodological* axis spans discrete DDPMs, continuous score-based SDEs,
conditional and guided variants, latent diffusion, fast-sampling schemes
(DDIM, consistency, flow matching), and structure-informed designs that bake in
financial priors such as geometric Brownian motion or factor structure. The
*task* axis spans (i) synthetic data and time-series generation, (ii) market
microstructure and limit-order-book (LOB) simulation, (iii) risk and scenario
generation, (iv) pricing, hedging, and calibration, (v) forecasting and
portfolio construction, and (vi) tabular data for credit, fraud, and privacy.

**Relation to existing surveys.** Prior reviews of generative models in finance
are GAN/VAE-centric; broad treatments of synthetic data in the GenAI era
\cite{li2025gensynthsurvey} are not finance-specific, and diffusion-focused
surveys to date target generic time series and spatio-temporal data
\cite{yang2024tssurvey} or anomaly detection \cite{liu2025anomalysurvey} rather
than finance specifically. To our knowledge this is the first survey
dedicated to diffusion models *in finance*, spanning data generation, market
simulation, pricing, risk, and portfolio decisions within a single taxonomy.

**Contributions.**
1. A two-axis taxonomy (method $\times$ financial task) that organizes a
   fast-growing and fragmented literature.
2. A consolidated account of how diffusion's conditioning and SDE structure are
   exploited for finance-specific goals—risk-neutral generation, stylized-fact
   fidelity, controllable stress scenarios, and microstructure realism.
3. A critical review of *evaluation*: stylized-fact checklists, distributional
   and downstream-task metrics, microstructure benchmarks, and privacy measures,
   together with the methodological pitfalls (leakage, look-ahead bias,
   in-distribution evaluation) that currently limit comparability.
4. A research agenda: tail fidelity, non-stationarity guarantees, standardized
   benchmarks, efficiency for real-time use, and regulatory/model-risk concerns.

The remainder of the paper proceeds as follows. Section 2 reviews diffusion
fundamentals and the bridge to financial mathematics. Sections 3–7 survey the
application areas. Section 8 treats evaluation and reproducibility. Section 9
discusses open problems, and Section 10 concludes.
