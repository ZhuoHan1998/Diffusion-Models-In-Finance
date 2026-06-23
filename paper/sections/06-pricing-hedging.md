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
