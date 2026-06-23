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
