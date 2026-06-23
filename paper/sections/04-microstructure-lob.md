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
