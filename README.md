# Awesome Diffusion Models in Finance Papers

A curated list of papers on learned generative diffusion, score-based, rectified-flow, flow-matching, and closely related stochastic-interpolant models for finance.

This repository also contains an ICAIF-style survey manuscript in [`paper/icaif/main.tex`](paper/icaif/main.tex) and an auditable companion corpus in [`paper/icaif/companion/corpus_coding.csv`](paper/icaif/companion/corpus_coding.csv).

Last updated: 2026-07-22.

## Scope

Included papers should use a learned generative diffusion/score/flow sampler, or make a finance-specific contribution that directly evaluates such models. Classical financial "diffusion" in the SDE/PDE sense is not included unless the paper also contains a learned generative diffusion or flow model.

Related time-series, tabular, decision-making, and non-diffusion finance generators are listed separately as background or baselines.

Some papers are cross-listed when they span more than one financial task, such as portfolio construction and stress-scenario generation.

Within each category, papers are ordered by year and then alphabetically by title.

## Contents

- [Surveys and Resources](#surveys-and-resources)
- [Synthetic Financial Time Series and Market Data](#synthetic-financial-time-series-and-market-data)
- [Limit Order Books and Market Simulation](#limit-order-books-and-market-simulation)
- [Forecasting, Portfolio Construction, and Trading](#forecasting-portfolio-construction-and-trading)
- [Risk, Scenario Generation, and Stress Testing](#risk-scenario-generation-and-stress-testing)
- [Pricing, Hedging, Volatility, and Fixed Income](#pricing-hedging-volatility-and-fixed-income)
- [Tabular, Fraud, Privacy, and Compliance](#tabular-fraud-privacy-and-compliance)
- [Flow Matching and Rectified Flow](#flow-matching-and-rectified-flow)
- [Related Foundations and Backbones](#related-foundations-and-backbones)
- [Non-Diffusion Finance Baselines](#non-diffusion-finance-baselines)
- [Contributing](#contributing)

## Surveys and Resources

- A Survey on Diffusion Models for Time Series and Spatio-Temporal Data, ACM Computing Surveys 2024. [paper](https://arxiv.org/abs/2404.18886) [awesome-list](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
- Diffusion Models for Tabular Data: Challenges, Current Progress, and Future Directions, 2024+. [awesome-list](https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data)
- Synthetic Data Applications in Finance, arXiv 2024. [paper](https://arxiv.org/abs/2401.00081)
- Deep Generative Models for Synthetic Financial Data: Applications to Portfolio and Risk Modeling, arXiv 2025. [paper](https://arxiv.org/abs/2512.21798)
- Generative Models for Synthetic Data: Transforming Data Mining in the GenAI Era, arXiv 2025. [paper](https://arxiv.org/abs/2508.19570)
- Diffusion Models in Finance: A Survey, 2026. [manuscript](paper/icaif/main.pdf) [source](paper/icaif/main.tex) [corpus](paper/icaif/companion/corpus_coding.csv)

## Synthetic Financial Time Series and Market Data

- Beyond Monte Carlo: Harnessing Diffusion Models to Simulate Financial Market Dynamics, arXiv 2024. [paper](https://arxiv.org/abs/2412.00036)
- Synthetic Asset Price Paths Generation Using Denoising Diffusion Probabilistic Model, SCML 2024. [event](https://www.scml.jp/scml2024.html)
- A diffusion-based generative model for financial time series via geometric Brownian motion, arXiv 2025. [paper](https://arxiv.org/abs/2507.19003)
- CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation, arXiv 2025. [paper](https://arxiv.org/abs/2503.04164)
- Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure, arXiv 2025. [paper](https://arxiv.org/abs/2504.06566)
- Diffusion Generative Models Meet Compressed Sensing, with Applications to Imaging and Finance, arXiv 2025. [paper](https://arxiv.org/abs/2509.03898)
- Fiaingen: A Financial Time Series Generative Method Matching Real-World Data Quality, arXiv 2025. [paper](https://arxiv.org/abs/2510.01169)
- Generation of Synthetic Financial Time Series by Diffusion Models, Quantitative Finance 2025. [paper](https://arxiv.org/abs/2410.18897)
- InterDiff: Synthesizing Financial Time Series with Inter-Stock Correlations via Classifier-Free Guided Diffusion, ECML-PKDD 2025. [paper](https://link.springer.com/chapter/10.1007/978-3-032-01655-9_13)
- High-Quality Synthetic Financial Time-Series using a GAN-Diffusion Framework, arXiv 2026. [paper](https://arxiv.org/abs/2605.27113)

## Limit Order Books and Market Simulation

- ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space, arXiv 2025. [paper](https://arxiv.org/abs/2508.02247)
- DiffVolume: Diffusion Models for Volume Generation in Limit Order Books, arXiv 2025. [paper](https://arxiv.org/abs/2508.08698)
- Painting the Market: Generative Diffusion Models for Financial Limit Order Book Simulation and Forecasting, arXiv 2025. [paper](https://arxiv.org/abs/2509.05107)
- Towards Realistic and Interpretable Market Simulations: Factorizing Financial Power Law using Optimal Transport, arXiv 2025. [paper](https://arxiv.org/abs/2507.09863)
- TRADES: Generating Realistic Market Simulations with Diffusion Models, arXiv 2025. [paper](https://arxiv.org/abs/2502.07071) [code](https://github.com/LeonardoBerti00/TRADES)
- Controllable Financial Market Generation with Diffusion Guided Meta Agent, AAAI 2026. [paper](https://arxiv.org/abs/2408.12991)
- DiffLOB: Diffusion Models for Counterfactual Generation in Limit Order Books, arXiv 2026. [paper](https://arxiv.org/abs/2602.03776)
- Limit Order Book Event Stream Prediction with Diffusion Model, Data Science and Engineering 2026. [paper](https://arxiv.org/abs/2412.09631) [journal](https://link.springer.com/article/10.1007/s41019-025-00328-4)

## Forecasting, Portfolio Construction, and Trading

- Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, CIKM 2023. [paper](https://dl.acm.org/doi/10.1145/3583780.3614915) [code](https://github.com/kentang2017/diffusion-vae)
- A Financial Time Series Denoiser Based on Diffusion Model, arXiv 2024. [paper](https://arxiv.org/abs/2409.02138)
- A Spatio-Temporal Diffusion Model for Missing and Real-Time Financial Data Inference, CIKM 2024. [paper](https://dl.acm.org/doi/10.1145/3627673.3679806) [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5104672)
- DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation, arXiv 2024. [paper](https://arxiv.org/abs/2402.06656)
- DiffSTOCK: Probabilistic Relational Stock Market Predictions using Diffusion Models, ICASSP 2024. [paper](https://arxiv.org/abs/2403.14063)
- DHMoE: Diffusion Generated Hierarchical Multi-Granular Expertise for Stock Prediction, AAAI 2025. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33250)
- Diffolio: A Diffusion Model for Multivariate Probabilistic Financial Time-Series Forecasting and Portfolio Construction, arXiv 2025. [paper](https://arxiv.org/abs/2511.07014)
- Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios, arXiv 2025. [paper](https://arxiv.org/abs/2510.07099)
- Exploring Diffusion Models for Generative Forecasting of Financial Charts, arXiv 2025. [paper](https://arxiv.org/abs/2509.02308)
- Factor-Based Conditional Diffusion Model for Contextual Portfolio Optimization, arXiv 2025. [paper](https://arxiv.org/abs/2509.22088)
- Multi-Agent Regime-Conditioned Diffusion (MARCD) for CVaR-Constrained Portfolio Decisions, arXiv 2025. [paper](https://arxiv.org/abs/2510.10807)
- Solving Dynamic Portfolio Selection Problems via Score-Based Diffusion Models, arXiv 2025. [paper](https://arxiv.org/abs/2507.09916)
- Factor Dimensionality and the Bias-Variance Tradeoff in Diffusion Portfolio Models, arXiv 2026. [paper](https://arxiv.org/abs/2603.10385)
- VARDiff: Vision-augmented retrieval-guided diffusion for stock forecasting, Information Sciences 2026. [paper](https://doi.org/10.1016/j.ins.2026.123113)

## Risk, Scenario Generation, and Stress Testing

- DiffCharge: Generating EV Charging Scenarios via a Denoising Diffusion Model, arXiv 2023. [paper](https://arxiv.org/abs/2308.09857)
- Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios, arXiv 2025. [paper](https://arxiv.org/abs/2510.07099)
- Diffusion-Based Scenario Tree Generation for Multivariate Time Series Prediction and Multistage Stochastic Optimization, arXiv 2025. [paper](https://arxiv.org/abs/2509.14832)
- Multi-Agent Regime-Conditioned Diffusion (MARCD) for CVaR-Constrained Portfolio Decisions, arXiv 2025. [paper](https://arxiv.org/abs/2510.10807)
- Neural Levy SDE for State-Dependent Risk and Density Forecasting, arXiv 2025. [paper](https://arxiv.org/abs/2509.01041)
- SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis, arXiv 2025. [paper](https://arxiv.org/abs/2509.22352)
- The Day-Ahead Scenario Generation Method for New Energy Based on an Improved Conditional Generative Diffusion Model, arXiv 2025. [paper](https://arxiv.org/abs/2503.07648)

## Pricing, Hedging, Volatility, and Fixed Income

- A Time-Stepping Deep Gradient Flow Method for Option Pricing in (Rough) Diffusion Models, arXiv 2024. [related neural-PDE paper](https://arxiv.org/abs/2403.00746)
- Forecasting Implied Volatility Surface with Generative Diffusion Models, arXiv 2025. [paper](https://arxiv.org/abs/2511.07571)
- Generating Solution Paths of Markovian Stochastic Differential Equations using Diffusion Models, arXiv 2025. [paper](https://arxiv.org/abs/2509.08731)
- Valuation of Exotic Options and Counterparty Games Based on Conditional Diffusion, arXiv 2025. [paper](https://arxiv.org/abs/2509.13374)
- Generating the Term Structure of Interest Rates with Diffusion Models, Expert Systems with Applications 2026. [paper](https://doi.org/10.1016/j.eswa.2026.132895)
- Generative Diffusion Model for Risk-Neutral Derivative Pricing, arXiv 2026. [paper](https://arxiv.org/abs/2603.20582)
- TF-CoDiT: Conditional Time Series Synthesis with Diffusion Transformers for Treasury Futures, arXiv 2026. [paper](https://arxiv.org/abs/2601.11880)

## Tabular, Fraud, Privacy, and Compliance

- FinDiff: Diffusion Models for Financial Tabular Data Generation, ICAIF 2023. [paper](https://arxiv.org/abs/2309.01472)
- Denoising Diffusion Probabilistic Model for Realistic Financial Correlation Matrices, ICAIF 2024. [paper](https://doi.org/10.1145/3677052.3698640) [record](https://dblp.org/rec/conf/icaif/KubiakWGPG24.html)
- Entity-based Financial Tabular Data Synthesis with Diffusion Models, ICAIF 2024. [paper](https://doi.org/10.1145/3677052.3698625)
- Federated Diffusion Modeling with Differential Privacy for Tabular Data Synthesis, arXiv 2024. [paper](https://arxiv.org/abs/2412.16083)
- Imb-FinDiff: Conditional Diffusion Models for Class Imbalance Synthesis of Financial Tabular Data, ICAIF 2024. [paper](https://doi.org/10.1145/3677052.3698659)
- Synthetic Data Generation for Fraud Detection Using Diffusion Models, Information and Security 2024. [paper](https://www.isij.eu/article/synthetic-data-generation-fraud-detection-using-diffusion-models)
- Boosting Predictive Performance on Tabular Data through Data Augmentation with Latent-Space Flow-Based Diffusion, arXiv 2025. [paper](https://arxiv.org/abs/2511.16571)
- Constrained Tabular Diffusion for Finance, ICAIF 2025 / arXiv 2026. [paper](https://arxiv.org/abs/2606.28674)
- Privacy Preserving Diffusion Models for Mixed-Type Tabular Data Generation, arXiv 2025. [paper](https://arxiv.org/abs/2512.00638)
- EmDT: Embedding Diffusion Transformer for Tabular Data Generation in Fraud Detection, arXiv 2026. [paper](https://arxiv.org/abs/2603.13566)
- Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation, arXiv 2026. [paper](https://arxiv.org/abs/2602.09288)
- Tabular diffusion counterfactual explanations, Frontiers in Artificial Intelligence 2026. [paper](https://doi.org/10.3389/frai.2026.1743495)

## Flow Matching and Rectified Flow

Finance-specific:

- Boosting Predictive Performance on Tabular Data through Data Augmentation with Latent-Space Flow-Based Diffusion, arXiv 2025. [paper](https://arxiv.org/abs/2511.16571)
- FlowHFT: Imitation Learning via Flow Matching Policy for Optimal High-Frequency Trading under Diverse Market Conditions, arXiv 2025. [paper](https://arxiv.org/abs/2505.05784)

General time-series/tabular flow papers useful for finance:

- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, ICLR 2023. [paper](https://arxiv.org/abs/2209.03003)
- Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, arXiv 2023. [paper](https://arxiv.org/abs/2303.08797)
- FlowTS: Time Series Generation via Rectified Flow, arXiv 2024. [paper](https://arxiv.org/abs/2411.07506)
- Probabilistic Forecasting with Stochastic Interpolants and Follmer Processes, ICML 2024. [paper](https://arxiv.org/abs/2403.13724)
- Recurrent Interpolants for Probabilistic Time Series Prediction, arXiv 2024. [paper](https://arxiv.org/abs/2409.11684)
- RecTable: Fast Modeling Tabular Data with Rectified Flow, arXiv 2025. [paper](https://arxiv.org/abs/2503.20731)
- TimeFlow: Towards Stochastic-Aware and Efficient Time Series Generation via Flow Matching Modeling, arXiv 2025. [paper](https://arxiv.org/abs/2511.07968)

## Related Foundations and Backbones

Diffusion and score-based modeling:

- Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015. [paper](https://arxiv.org/abs/1503.03585)
- Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019. [paper](https://arxiv.org/abs/1907.05600)
- Denoising Diffusion Probabilistic Models, NeurIPS 2020. [paper](https://arxiv.org/abs/2006.11239)
- Denoising Diffusion Implicit Models, ICLR 2021. [paper](https://arxiv.org/abs/2010.02502)
- Diffusion Models Beat GANs on Image Synthesis, NeurIPS 2021. [paper](https://arxiv.org/abs/2105.05233)
- Improved Denoising Diffusion Probabilistic Models, ICML 2021. [paper](https://arxiv.org/abs/2102.09672)
- Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 2021. [paper](https://arxiv.org/abs/2011.13456)
- Classifier-Free Diffusion Guidance, NeurIPS Workshop 2022. [paper](https://arxiv.org/abs/2207.12598)
- Elucidating the Design Space of Diffusion-Based Generative Models, NeurIPS 2022. [paper](https://arxiv.org/abs/2206.00364)
- Consistency Models, ICML 2023. [paper](https://arxiv.org/abs/2303.01469)

Time-series and tabular backbones:

- Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting, ICML 2021. [paper](https://arxiv.org/abs/2101.12072)
- CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, NeurIPS 2021. [paper](https://arxiv.org/abs/2107.03502)
- Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models, TMLR 2022. [paper](https://arxiv.org/abs/2208.09399)
- Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting, NeurIPS 2023. [paper](https://arxiv.org/abs/2307.11494)
- TabDDPM: Modelling Tabular Data with Diffusion Models, ICML 2023. [paper](https://arxiv.org/abs/2209.15421)
- Diffusion-TS: Interpretable Diffusion for General Time Series Generation, ICLR 2024. [paper](https://arxiv.org/abs/2403.01742)
- Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space, ICLR 2024. [paper](https://arxiv.org/abs/2310.09656)
- DS-Diffusion: Data Style-Guided Diffusion Model for Time-Series Generation, arXiv 2025. [paper](https://arxiv.org/abs/2509.18584)

Decision-making and control:

- Planning with Diffusion for Flexible Behavior Synthesis, ICML 2022. [paper](https://arxiv.org/abs/2205.09991)
- Diffusion Models for Reinforcement Learning: A Survey, arXiv 2023. [paper](https://arxiv.org/abs/2311.01223)
- Is Conditional Generative Modeling all you need for Decision-Making?, ICLR 2023. [paper](https://arxiv.org/abs/2211.15657)
- An Optimal Control Perspective on Diffusion-Based Generative Modeling, TMLR 2024. [paper](https://arxiv.org/abs/2211.01364)

## Non-Diffusion Finance Baselines

These are useful comparison points, but they are not counted as diffusion/flow finance papers.

- Generalized Autoregressive Conditional Heteroskedasticity, Journal of Econometrics 1986. [paper](https://doi.org/10.1016/0304-4076(86)90063-1)
- A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options, Review of Financial Studies 1993. [paper](https://doi.org/10.1093/rfs/6.2.327)
- Hawkes Processes in Finance, Market Microstructure and Liquidity 2015. [paper](https://doi.org/10.1142/S2382626615500057)
- Volatility is Rough, Quantitative Finance 2018. [paper](https://doi.org/10.1080/14697688.2017.1393551)
- Deep Hedging, Quantitative Finance 2019. [paper](https://arxiv.org/abs/1802.03042)
- Quant GANs: Deep Generation of Financial Time Series, Quantitative Finance 2020. [paper](https://arxiv.org/abs/1907.06673)
- Sig-Wasserstein GANs for Time Series Generation, ICAIF 2021. [paper](https://arxiv.org/abs/2111.01207)
- Generative Modelling of Financial Time Series with Structured Noise and MMD-based Signature Learning, arXiv 2024. [paper](https://arxiv.org/abs/2407.19848)
- Bayesian Robust Financial Trading with Adversarial Synthetic Market Data, arXiv 2026. [paper](https://arxiv.org/abs/2601.17008)

## Contributing

Please open an issue or pull request if a paper is missing, miscategorized, or no longer current.

Preferred entry format:

```markdown
- Paper Title, Venue Year. [paper](https://...) [code](https://...) - optional short note.
```

Before adding a paper, check:

- The model is a learned generative diffusion, score-based, flow-matching, rectified-flow, or stochastic-interpolant model.
- The application is financial, or the paper is clearly useful as a backbone/baseline for finance.
- The title, authors, year, venue, and paper link resolve to a real source.
- Classical SDE/PDE "diffusion" papers are not added unless they contain a learned generative sampler.

## Repository Layout

```text
state/    framework state, findings, progress, and iteration logs
logs/     work, orchestrator, and heartbeat logs
paper/    comprehensive survey draft, ICAIF paper, references, and companion corpus
```
