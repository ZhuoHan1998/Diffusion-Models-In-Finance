# Diffusion Models in Finance: A Survey

_Comprehensive draft (Deli_AutoResearch). Build-big-then-condense toward ICAIF ACM sigconf._

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


# 4. Market Microstructure and Limit-Order-Book Simulation

Simulating the limit order book (LOB) and the order flow that drives it is a
distinct and harder problem than generating low-frequency returns. The data are
high-frequency, discrete, bursty, and shaped by exchange rules; a useful
simulator must also be *responsive*—reacting plausibly to an agent's own
orders—so that it can serve strategy backtesting, market-impact studies, and
agent-based experiments. Diffusion models have rapidly become a leading approach.

## 4.1 Generating order flow

TRADES \cite{berti2025trades} (a transformer-based denoising diffusion engine)
generates realistic order flows as a time series conditioned on the market state,
capturing temporal and cross-sectional structure of high-frequency data. The
authors report sizable improvements in predictive score over prior simulators on
multiple stocks, and position the model as a substrate for agent-based
simulation. Order-flow generation is attractive because the LOB can be
reconstructed deterministically from the flow, so realism in the flow propagates
to realism in the book.

## 4.2 Image-based and inpainting representations

A complementary strategy renders the LOB as a structured image and applies
diffusion with *inpainting*: \citet{backhouse2025painting} generate future LOB
states by conditioning on the observed past, exploiting spatio-temporal inductive
biases and generating long horizons in parallel. This sidesteps the error
accumulation that plagues autoregressive simulators and reports strong results on
LOB-Bench (§8). The image framing echoes the wavelet-to-image idea in §3.1: when
the representation encodes the right structure, generic diffusion machinery
transfers well.

## 4.3 Volume and depth generation

Beyond price levels, the *volume* profile across the book governs liquidity and
market impact. DiffVolume \cite{wang2025diffvolume} introduces a conditional
diffusion model for LOB volume snapshots, conditioned on historical order flow
and time, and—critically—supports *counterfactual* generation: volume under
hypothetical market conditions. This controllability makes the model usable for
scenario-based liquidity analysis and what-if stress testing at the
microstructure level, linking microstructure simulation to the risk theme of §5.

## 4.4 Controllable, responsive market generation

Controllability and responsiveness are the frontier. DigMA \cite{huang2024digma}
pairs a conditional diffusion model of the market state with a financially
informed meta-agent, enabling controllable generation of market orders while
preserving realistic dynamics, with demonstrated utility for high-frequency
trading. At lower frequency, \citet{lesniewski2024beyondmc} use diffusion to
simulate market dynamics "beyond Monte Carlo," producing regularized covariance
matrices suitable for portfolio analysis—a bridge from microstructure realism to
portfolio-level objects (§7).

## 4.5 Evaluation of market simulators

Microstructure realism resists single-number summaries. Benchmarks such as
LOB-Bench \cite{backhouse2025painting} aggregate distributional and
predictive criteria, while interpretability-oriented work uses optimal transport
to test whether a simulator reproduces the empirical power law of returns and to
attribute it to mechanism components \cite{hashimoto2025otpowerlaw}. Open issues
include measuring *responsiveness* (impact realism under the simulated agent's own
actions) and avoiding look-ahead leakage; we treat evaluation systematically in
§8.

## 4.6 Counterfactual generation, event streams, and raw-byte models

Three further threads round out the microstructure picture. *Counterfactual*
generation makes the simulator answer "what-if": DiffLOB \cite{wang2026difflob}
is a regime-conditioned diffusion for controllable, counterfactual LOB
trajectories, complementing the volume-level counterfactuals of DiffVolume
\cite{wang2025diffvolume}. *Event-stream* models predict the timing and type of
LOB events jointly: LOBDIF \cite{zheng2024lobdif} casts this as a diffusion over
the time–event distribution with a skip-step sampler for high-frequency latency.
And *representation* continues to be a design frontier—ByteGen \cite{li2025bytegen}
forgoes tokenization entirely, generating order-book events directly in byte
space with a hybrid Mamba–Transformer. Diffusion is also used upstream as a
*denoiser* of noisy market series to improve trading and classification
downstream \cite{wang2024ftsdenoiser}, a complementary role to generation.


# 5. Risk, Scenario Generation, and Stress Testing

Risk management is where the controllability of diffusion models pays off most
directly. The task is not to reproduce the average market but to populate the
*tails* and the *what-ifs*: crashes, liquidity droughts, and regime breaks that
are rare or absent in the historical record. Conditioning lets a single trained
model act as a scenario generator steered toward specified stress.

## 5.1 Conditional tail and crash scenarios

Because diffusion models admit attribute conditioning via guidance
\cite{dhariwal2021beatgan, ho2022cfg}, they can be asked for samples *given* a
crash intensity, a volatility regime, or a macro state, generating tail
scenarios that are under-represented in training data. The same heavy-tail
fidelity that makes diffusion attractive for data generation (§3.1) is what makes
the resulting Value-at-Risk and Expected-Shortfall estimates credible: a
generator that systematically thins tails will understate risk. This places a
premium on the stylized-fact evaluation of §8 whenever a model is used for risk.
Many of the conditioning techniques used here transfer from adjacent
scenario-generation domains—e.g., energy and mobility scenario synthesis such as
EV-charging profiles \cite{li2023diffcharge}—where conditional diffusion has been
validated before adaptation to financial risk.

## 5.2 Stress scenarios coupled to decisions

Scenario generation is most useful when wired into a decision. \citet{choudhary2025diffaugrl}
combine a diffusion model that synthesizes market-stress scenarios with deep
reinforcement learning, training portfolio policies that remain robust under the
generated stress—an explicit pipeline from generative stress to robust action.
This *generate-then-optimize* pattern recurs in portfolio construction (§7) and
is one of the more promising templates for operational use.

## 5.3 Structure-aware scenario generation

High-dimensional risk requires realistic *dependence*, not just realistic
marginals. The factor-structured diffusion of \citet{chen2025difffactor}
generates joint return scenarios that preserve a factor covariance, allowing
stress to be applied at the factor level (e.g., a value or momentum shock) while
keeping cross-sectional dependence coherent. At the microstructure level,
counterfactual volume generation \cite{wang2025diffvolume} supplies
liquidity-stress scenarios—book states under hypothetical conditions—so that
execution and impact risk can be probed, not only price risk.

## 5.4 From simulation to regulated stress testing

Regulatory stress testing (e.g., supervisory capital exercises) traditionally
relies on a small number of hand-designed scenarios. Diffusion-based generation
offers a route to richer, internally consistent scenario sets, but raises its own
governance questions: a scenario generator used for capital must itself be
validated, auditable, and stable under non-stationarity. We flag these
model-risk concerns here and develop them in §9; they are, in our view, the main
barrier between the research results above and regulated deployment.

## 5.5 Summary

Across risk applications the value proposition is consistent—controllable,
dependence-aware tail scenarios on demand—and the central caveat is equally
consistent: a risk number is only as trustworthy as the generator's tail
fidelity and its robustness to the regime change it is meant to anticipate.

## 5.6 Tail-aware generation, regimes, and decision pipelines

Recent risk work converges on *tail-* and *regime-aware* generation wired
directly to decisions. MARCD \cite{alzahrani2025marcd} couples a regime-aligned
diffusion generator (via a hidden Markov model) to a CVaR-constrained allocator
with a tail-weighted objective, reporting large drawdown reductions across two
decades of ETF data. Compressed-sensing-accelerated diffusion in a low-dimensional
macro factor space reproduces left-tail quantiles for backtesting
\cite{guo2025compressedsensing}, and diffusion *scenario trees*
\cite{zarifis2025scenariotree} feed multistage stochastic optimization. On the
modeling side, neural Lévy jump-diffusion \cite{wang2025neuralllevy} targets
state-dependent risk and density forecasting where jumps dominate tails. Many of
these inherit conditional-scenario techniques validated in energy markets
\cite{wang2025dayaheadenergy}.


# 6. Pricing, Hedging, and Calibration

Pricing is the application where diffusion models meet the hardest economic
constraints. Unlike data generation, where realism is the goal, pricing demands
*no-arbitrage* and the correct change of measure from the physical ($\mathbb{P}$)
to the risk-neutral ($\mathbb{Q}$) world. This section reviews how diffusion and
score-based methods are being adapted to honor that structure.

## 6.1 The risk-neutral generation problem

A generator trained on historical paths learns the physical measure, but
derivative prices are expectations under $\mathbb{Q}$. Two routes appear in the
literature. The first builds the change of measure into the generator's
construction, as in the risk-neutral transition designed into Quant GANs
\cite{wiese2020quantgan}; the analogous program for diffusion exploits the fact
that a change of measure shifts the drift of the underlying SDE
\cite{song2021sde}, which in score/$\epsilon$ terms is an additive correction to
the learned reverse dynamics. The second route calibrates a conditional
generator directly to observed market prices. Both inherit diffusion's ability to
represent rich, non-Gaussian terminal distributions while remaining tied to a
tractable SDE.

## 6.2 Exotic and path-dependent options

\citet{zhao2025exotic} use conditional diffusion to value exotic options and
counterparty games, reporting accurate pricing of European and path-dependent
payoffs (e.g., arithmetic Asian options) by sampling realistic terminal and path
distributions. The honest limitation reported—degraded accuracy on *extreme*
scenarios—mirrors the tail-fidelity theme of §5 and is the crux for risk-sensitive
products.

## 6.3 Implied-volatility surfaces

The implied-volatility (IV) surface is a structured, constrained object:
arbitrage imposes shape restrictions across strike and maturity.
\citet{jin2025ivsurface} generate arbitrage-free IV surfaces one day ahead with a
conditional DDPM, using a signal-to-noise-ratio weighting that penalizes
arbitrage violations—an instance of encoding an economic constraint as a training
penalty rather than an architectural hard constraint. This is more stable and
accurate than GAN-based surface models in their experiments and is directly
useful for option risk and calibration.

## 6.4 Diffusion for SDEs, PDEs, and stochastic control

A more foundational line treats diffusion models as solvers for the stochastic
objects pricing rests on. \citet{gao2025sdepaths} generate solution paths of
Markovian SDEs with a conditional diffusion model *without* knowing the drift and
diffusion coefficients—learning to simulate an unknown dynamics from data, which
is exactly the calibration setting when the true model is unknown. The
control-theoretic reading of diffusion training as a Hamilton–Jacobi–Bellman
problem on log-densities \cite{berner2022optcontrol} connects this to the HJB
equations of optimal investment and hedging, suggesting diffusion samplers as
tools for high-dimensional stochastic-control problems in finance.

## 6.5 Hedging

Deep hedging \cite{buehler2019deephedging}—learning a hedging policy directly by
neural optimization of a risk measure under frictions—is the downstream decision
that generative models serve, since its learned hedge is only as good as the
distribution of market paths it trains on. Controllable generators feed it:
CoFinDiff \cite{tanaka2025cofindiff} reports improved hedging performance when its
controllable synthetic paths are used, illustrating the indirect but practical
route by which generation improves hedging—better training distributions yield
better learned hedges. Direct diffusion-based hedging policies remain an open
direction (§9).

## 6.6 Summary

Diffusion methods bring expressive, SDE-grounded distributions to pricing, with
early successes on exotics and IV surfaces. The unresolved frontier is enforcing
*hard* economic constraints (exact no-arbitrage, exact martingale property under
$\mathbb{Q}$) rather than penalizing their violation, and doing so while
preserving tail accuracy.

## 6.7 Risk-neutral pricing, rates, and neural PDE solvers

Two developments sharpen the pricing picture. First, the risk-neutral generation
problem of §6.1 now has a direct treatment: \citet{tiwari2026riskneutral} apply a
score-function shift to DDPM dynamics that enforces the martingale (risk-neutral)
condition, making the change of measure explicit rather than approximate. Second,
diffusion methods have reached *fixed income*: conditional diffusion generates
arbitrage-aware yield-curve term structures with macroeconomic conditioning
\cite{fukunishi2025termstructure}, and diffusion transformers synthesize
language-conditioned Treasury-futures series \cite{zhang2026tfcodit}. These sit
alongside neural solvers for the pricing PDEs of (rough) volatility models
\cite{papapantoleon2024deepgradflow}—not generative diffusion, but a
complementary deep-learning route to the same valuation problem.


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


# 8. Evaluation, Benchmarks, and Reproducibility

Evaluation is the weakest link in this literature, and getting it right matters
more in finance than in vision: a generator that looks good by a generic metric
can be useless—or dangerous—if it misprices tail risk or leaks look-ahead
information. We organize evaluation into four layers.

## 8.1 Stylized-fact fidelity

The first-line test is whether generated data reproduces the canonical stylized
facts: heavy tails, volatility clustering, the leverage effect, and the
autocorrelation signatures of returns versus absolute returns. Concrete measures
include excess kurtosis and the Hill tail-index estimator for fat tails, and
autocorrelation-function comparisons for clustering
\cite{takahashi2024synthfts, kim2025gbmdiff}. Optimal-transport distances provide
a more interpretable, mechanism-attributing alternative for market simulators
\cite{hashimoto2025otpowerlaw}. Stylized-fact checks are necessary but not
sufficient—they are marginal and largely univariate.

## 8.2 Distributional and downstream metrics

Beyond marginals, two families dominate. *Discriminative/predictive scores* train
a classifier to distinguish real from synthetic (lower is better) or measure how
well a model trained on synthetic data predicts real data; market simulators such
as TRADES \cite{berti2025trades} report predictive-score gains. *Train-on-
synthetic, test-on-real* (TSTR) is the operational standard: Fiaingen
\cite{rozanec2025fiaingen} reports near-real downstream performance, which is the
metric that actually justifies replacing real data. Both are more meaningful than
visual or marginal checks because they probe *joint* and *task-relevant* structure.

## 8.3 Microstructure and task-specific benchmarks

For LOB simulation, LOB-Bench \cite{backhouse2025painting} aggregates
distributional and predictive criteria over book and flow. What remains poorly
standardized is *responsiveness*—whether a simulator reacts realistically to the
evaluated agent's own orders—which is exactly the property needed for strategy
backtesting and market-impact studies, yet is not captured by static
distributional scores.

## 8.4 Privacy

When synthetic data substitutes for sensitive records, privacy is a first-class
metric. FinDiff \cite{sattarov2023findiff} evaluates privacy alongside fidelity
and utility, reporting advantages over TVAE/CTGAN/TabDDPM on financial tables.
Moving from empirical proxies toward *formal* guarantees, DP-FinDiff
\cite{sattarov2025dpfindiff} and the federated DP-FedTabDiff
\cite{sattarov2024dpfedtabdiff} train diffusion models under differential privacy,
trading utility against a stated privacy budget, while \citet{zuo2026privacyrisks}
benchmark the privacy–utility frontier across autoencoder, GAN, diffusion, and
copula synthesizers on financial data. Membership-inference resistance and formal
differential privacy are the right targets; most non-DP work still reports
empirical proxies only.

## 8.5 Reproducibility and methodological pitfalls

Three problems recur and undermine cross-paper comparison. (i) *Heterogeneous
datasets and metrics*: papers use different assets, periods, and scores, so
reported superiority rarely transfers. (ii) *Leakage and look-ahead bias*:
normalization or imputation computed over the full sample, or conditioning on
information unavailable at decision time, inflates results—especially in
forecasting and portfolio studies (§7). (iii) *In-distribution evaluation*:
testing on the same regime as training masks the non-stationarity failure that
matters in deployment. We recommend chronological splits, regime-shifted test
sets, fixed public benchmarks, and released code as minimum reproducibility
standards. The absence of a shared, finance-specific benchmark suite is, in our
assessment, the single biggest obstacle to measurable progress in this field.


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


# 10. Conclusion

Diffusion and score-based generative models have, in a short span, become a
general-purpose tool across quantitative finance. We surveyed their use along two
axes—method (DDPM, score-SDE, conditional/guided, latent, fast-sampling,
structure-informed) and task (synthetic data, microstructure simulation, risk and
scenario generation, pricing and hedging, forecasting and portfolio
construction, and tabular/credit/fraud applications). The throughline is that
diffusion's two defining properties—stable training with strong mode coverage,
and native, flexible conditioning—map unusually well onto financial needs, while
its SDE formulation sits naturally beside the Itô calculus, Monte Carlo, and
stochastic control that the field already speaks.

The evidence is genuinely encouraging: diffusion models reproduce stylized facts
more reliably than GAN/VAE predecessors, generate controllable stress scenarios,
simulate limit order books with reduced error accumulation, produce
dependence-aware probabilistic forecasts that improve portfolios, and price
exotic options and volatility surfaces under economic constraints. Yet the
field's progress is harder to measure than to produce. Tail fidelity degrades
exactly where finance cares most; evaluation is fragmented and frequently
in-distribution; economic constraints are penalized rather than enforced;
sampling is too slow for real-time use; and governance for regulated deployment
is barely begun.

Our assessment is that the next phase of progress depends less on new
architectures than on *discipline*: finance-specific benchmarks with
chronological and regime-aware splits, tail- and arbitrage-aware objectives,
formal privacy and model-risk treatment, and tighter coupling of generation to
the decision it serves. Diffusion models have earned a permanent place in the
quantitative-finance toolkit; turning that foothold into dependable practice is
the work ahead.


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


# Appendix A. Datasets and Benchmarks

A recurring obstacle to comparability (§8) is dataset heterogeneity. We summarize
the data sources and benchmarks that appear across the surveyed work, grouped by
task, to help practitioners choose appropriate evaluation settings.

## A.1 Low-frequency price/return series
Most synthetic-generation studies use daily equity index or single-name returns
(e.g., S&P 500 constituents and the index itself), sometimes with volume and
spread channels \cite{takahashi2024synthfts}. Cross-sectional studies use panels
of returns with a factor interpretation \cite{chen2025difffactor}; portfolio
studies use industry-portfolio return panels \cite{cho2025diffolio}. Evaluation
centers on stylized-fact statistics (kurtosis, Hill index, ACF of returns vs.
absolute returns) and downstream portfolio metrics.

## A.2 High-frequency / limit-order-book data
LOB studies use message-level and snapshot data for liquid equities, typically
from exchange feeds (e.g., NASDAQ-style ITCH messages) reconstructed into book
states and order flow \cite{berti2025trades, backhouse2025painting,
wang2025diffvolume, huang2024digma}. **LOB-Bench** \cite{backhouse2025painting}
is emerging as a shared evaluation suite combining distributional and predictive
criteria; responsiveness/impact realism remains non-standardized.

## A.3 Tabular financial records
Tabular studies use mixed-type records spanning consumer credit risk, public-
sector expenditure, and institutional fund composition \cite{sattarov2023findiff},
with fraud/credit-default datasets for the imbalanced-classification use case.
Privacy studies add medical tables for cross-domain comparison
\cite{sattarov2025dpfindiff, sattarov2024dpfedtabdiff, zuo2026privacyrisks}.
Metrics combine fidelity, downstream utility, and privacy (membership-inference
proxies, DP budgets).

## A.4 Options and rates
Pricing studies use option underlyings and payoff structures (European, Asian,
and other path-dependent) \cite{zhao2025exotic}, and implied-volatility surfaces
across strike/maturity grids \cite{jin2025ivsurface}. SDE-path studies use
synthetic SDEs with known ground truth for controlled evaluation
\cite{gao2025sdepaths}.

## A.5 Cross-domain scenario data
Energy and mobility scenario generators (e.g., EV charging profiles
\cite{li2023diffcharge}) are methodologically close to financial scenario
generation and are sometimes used to validate diffusion scenario techniques
before transfer to finance.

## A.6 Recommended reporting standard
To make results comparable, we recommend each study report: (i) exact data source,
asset universe, and date range; (ii) a **chronological** train/test split with an
explicit out-of-regime test period; (iii) at least one stylized-fact metric, one
discriminative/predictive score, and one downstream-task (TSTR) metric; (iv) for
private data, a stated privacy notion and budget; and (v) released code and
preprocessing. None of these is universal today.


