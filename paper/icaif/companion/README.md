# Companion Coding Artifact

This folder is the working companion artifact for the ICAIF survey draft. It is
intended to support anonymous review and later camera-ready release.

Files:

- `corpus_coding.csv`: recorded candidate and included works from the survey
  workflow, with task/method labels, modality, conditioning, evaluation notes,
  publication status, and code/dataset availability fields.

Search protocol:

- Sources: arXiv, SSRN, ACM Digital Library, IEEE Xplore, Google Scholar, and
  citation snowballing from included papers.
- Date cutoff: 2026 Q1.
- Method terms: `diffusion`, `score-based`, `flow matching`, `rectified flow`,
  `stochastic interpolant`.
- Finance terms: `market`, `limit order book`, `volatility`, `portfolio`,
  `risk`, `option pricing`, `tabular financial data`.
- Screening rule: retain works that apply a diffusion-family generator or
  generative decision model to a financial task; exclude generic time-series,
  vision, or non-diffusion papers without a financial diffusion instantiation.

Coding rules:

- `inclusion_status=included_or_context` means the record was retained as a cited
  work or necessary contextual baseline/foundation.
- `inclusion_status=replaced_candidate` means an early candidate record was later
  superseded by a verified record.
- `inclusion_status=excluded` means the work was screened out because it was not a
  diffusion-family method in scope.
- Code and dataset availability remain `not_verified` unless a link has been
  checked directly.
- The paper's coverage map uses the primary task and method labels; multi-use
  papers are assigned to the dominant reported application to avoid double
  counting.
