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
