# 2. Background: Diffusion and Score-Based Generative Models

This section gives the minimum machinery needed to read the rest of the survey,
and—more importantly—draws the bridge to the mathematical apparatus of
quantitative finance. Readers familiar with diffusion models may skim to §2.7.

## 2.1 The forward and reverse processes

A diffusion model defines a *forward* process that gradually corrupts data
$x_0 \sim p_{\text{data}}$ into noise, and learns a *reverse* process that
reconstructs data from noise. In the DDPM formulation \cite{ho2020ddpm}, the
forward process is a Markov chain that adds Gaussian noise over $T$ steps,
$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$,
admitting the closed form $x_t = \sqrt{\bar\alpha_t}\,x_0 +
\sqrt{1-\bar\alpha_t}\,\epsilon$. A network $\epsilon_\theta(x_t,t)$ is trained
to predict the injected noise via the simple denoising objective
$\mathbb{E}_{x_0,\epsilon,t}\big[\lVert \epsilon - \epsilon_\theta(x_t,t)\rVert^2\big]$.
The idea traces to non-equilibrium thermodynamics \cite{sohldickstein2015};
subsequent refinements add learned reverse variances and improved noise schedules
\cite{nichol2021improved}.

## 2.2 Score matching

Score-based models \cite{song2019ncsn} instead learn the *score*
$\nabla_x \log p_t(x)$ by denoising score matching across a range of noise
scales, and sample with Langevin dynamics. Predicting the noise and predicting
the score are equivalent up to scaling, so DDPMs and score models are two views
of one object.

## 2.3 The SDE unification

\citet{song2021sde} unify both under a continuous-time Itô SDE
$dx = f(x,t)\,dt + g(t)\,dW_t$, whose time reversal
$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar W_t$
is itself an SDE driven by the score. Every reverse SDE has a deterministic
*probability-flow ODE* with the same marginals, enabling exact-likelihood and
fast deterministic sampling. **This is the conceptual heart of the finance
connection:** asset-price models are SDEs, so a diffusion generator is a learned,
data-driven SDE, and sampling is the same reverse-time simulation quants already
perform—now with a neural drift in place of a hand-specified one.

## 2.4 Sampling and acceleration

Naive ancestral sampling needs hundreds to thousands of network evaluations,
which is prohibitive for high-frequency or large-scale risk applications.
DDIM \cite{songddim2021} introduces a non-Markovian deterministic sampler;
EDM \cite{karras2022edm} clarifies the design space (noise schedule,
preconditioning, sampler); consistency models \cite{song2023consistency} and
flow matching \cite{lipman2023flow} push toward one- or few-step generation.
Sampling cost is a recurring constraint in the finance applications below (§4, §9).

## 2.4b Flows and interpolants: the broader family

Flow matching is more than a fast sampler—it is the entry point to a
*generalization* of diffusion that this survey treats as in scope. Rectified flow
\cite{liu2023rectifiedflow} learns a neural ODE that transports noise to data
along (near-)straight paths, trained by simple least-squares regression of a
velocity field; stochastic interpolants \cite{albergo2023interpolants} give a
unifying framework in which a tunable interpolation between any two densities
yields *both* a probability-flow ODE and a family of SDEs, with classical
diffusion recovered as the Gaussian special case. The practical upshot for
finance is threefold: (i) the velocity-regression objective is simulation-free and
stable, like the diffusion loss; (ii) arbitrary base distributions and
interpolation schedules allow financial priors to be built into the *path*, not
just the prior; and (iii) the deterministic ODE form supports few-step sampling
where latency binds (§4). We therefore include flow-matching/rectified-flow/
interpolant methods alongside DDPM and score-SDE. We treat *classical* normalizing
flows (invertible, exact-likelihood maps) as a related baseline rather than as
diffusion, since their mechanism differs.

## 2.5 Conditioning and guidance

Finance rarely wants unconditional samples; it wants samples *given* a regime, a
volatility target, a stress label, or a history. Classifier guidance
\cite{dhariwal2021beatgan} and classifier-free guidance \cite{ho2022cfg} steer
generation toward desired attributes, and latent diffusion \cite{rombach2022ldm}
moves the process into a learned latent space for efficiency. These mechanisms
underpin controllable scenario generation (§5) and controllable market
simulation (§3, §4).

## 2.6 Diffusion for sequences and tables

Several cross-domain models recur as finance backbones and baselines. For
sequences: TimeGrad \cite{rasul2021timegrad}, an autoregressive diffusion for
multivariate probabilistic forecasting; CSDI \cite{tashiro2021csdi}, conditional
score-based imputation; SSSD \cite{alcaraz2022sssd}, which pairs diffusion with
structured state-space backbones; Diffusion-TS \cite{yuan2024diffusionts}, an
interpretable encoder–decoder transformer with a Fourier loss; and TSDiff
\cite{kollovieh2023tsdiff}, whose self-guidance adapts one unconditional model to
forecasting, refinement, and synthesis. For tables, TabDDPM
\cite{kotelnikov2023tabddpm} handles mixed numerical/categorical data via combined
Gaussian and multinomial diffusion—directly relevant to financial records (§3.3,
§6). Architecturally, the transformer backbone (DiT \cite{peebles2022dit}) now
underlies several financial models (TRADES, Diffolio). A parallel line treats
*decision-making itself* as conditional generation—Diffuser
\cite{janner2022diffuser}, Decision Diffuser \cite{ajay2022decisiondiffuser}, and
the broader diffusion-for-RL program \cite{zhu2023diff4rl}—which we revisit for
trading and portfolio policies (§7, §9.5). The flow/interpolant family supplies
its own rapidly growing set of sequence and table backbones: FlowTS
\cite{hu2024flowts} and TimeFlow \cite{he2025timeflow} generate time series via
rectified flow and flow matching respectively; stochastic-interpolant forecasting
\cite{chen2024siforecasting} and recurrent interpolants
\cite{chen2024recurrentinterpolants} bring the interpolant view to probabilistic
time-series prediction; and RecTable \cite{fuchi2025rectable} is the rectified-flow
analogue of TabSyn for tables. These are the natural backbones for the financial
flow-matching applications surveyed below.

## 2.7 Why finance is hard

Generic diffusion successes do not transfer automatically, because financial
data violate the assumptions under which image diffusion thrives:

- **Stylized facts.** Returns exhibit heavy (power-law) tails, volatility
  clustering, the leverage effect, and weak autocorrelation of returns but
  strong autocorrelation of absolute returns. A model can attain low pixel-style
  reconstruction error yet miss all of these
  \cite{takahashi2024synthfts, kim2025gbmdiff, hashimoto2025otpowerlaw}.
- **Non-stationarity and regime shifts.** The target distribution drifts; the
  most decision-relevant samples (crises) are under-represented in training data.
- **Low signal-to-noise ratio.** Predictable structure is faint relative to
  noise, so over-flexible generators can hallucinate spurious structure.
- **Microstructure constraints.** LOB and order-flow data are discrete, bursty,
  and governed by exchange rules; realism requires respecting these constraints
  (§4).
- **Economic consistency.** Pricing applications require no-arbitrage and the
  correct change of measure to the risk-neutral world (§6)—constraints with no
  analogue in vision.

These tensions motivate the *structure-informed* diffusion designs that recur
throughout the survey: GBM-based forward processes \cite{kim2025gbmdiff}, factor
structure \cite{chen2025difffactor}, arbitrage penalties \cite{jin2025ivsurface},
and risk-neutral drift corrections (§6). The remainder of the paper is organized
by financial task, with these methodological threads reappearing in each.
