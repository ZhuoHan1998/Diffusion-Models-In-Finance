# Diffusion Models in Finance — Survey

A comprehensive, citation-grounded survey of diffusion-based / score-based / flow-based generative models applied to finance, built toward an **ICAIF** submission.

Run under the **Deli_AutoResearch** protocol
(<https://victorchen96.github.io/auto_research/framework.html>): build a
comprehensive survey first, then condense to a ~8-page ACM `sigconf` paper.

## Layout
```
state/    framework state (task_spec, progress, findings, directions, iteration log)
logs/     work / orchestrator / heartbeat logs (jsonl)
research/ raw literature notes, verified reference data
paper/    outline.md, sections/, references.bib, comprehensive + ICAIF drafts
```

## Status
See `state/progress.json` for the live iteration/phase. Findings accumulate in
`state/findings.jsonl`; every reference is verified against a real source before use.
