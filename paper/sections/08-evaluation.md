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
