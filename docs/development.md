---
layout: default
title: Development
permalink: /development/
description: Testing, benchmarking, and maintaining fast-ballmapper documentation.
---

# Development

The repository keeps construction algorithms, range-query backends, and numerical validation separate. Start with the [function reference]({{ '/reference/' | relative_url }}) if you are trying to locate an implementation.

## Development installation

```bash
git clone https://github.com/jhnrckmnznrs/fast-ballmapper.git
cd fast-ballmapper
python -m pip install -e ".[dev,plot]"
```

Run the checks:

```bash
python -m ruff check .
pytest
python experiments/run_exactness_gate.py --smoke
```

## Experiment scripts

| Script | Purpose |
|---|---|
| [`run_paper_experiments.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/experiments/run_paper_experiments.py) | Main manuscript experiments |
| [`run_exactness_gate.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/experiments/run_exactness_gate.py) | Reference comparison for exact radius queries |
| [`benchmark_dispatch_and_completion.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/experiments/benchmark_dispatch_and_completion.py) | Exact cover and graph-builder comparisons |
| [`benchmark_budget_certificates.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/experiments/benchmark_budget_certificates.py) | Robustness solver comparisons |

A small smoke experiment:

```bash
python experiments/run_paper_experiments.py --smoke --output experiments/results/smoke
```

The experiment runner stores `manifest.json`, `summary.csv`, and `validation.json` for reproducibility. See each script's `--help` for available options.

## Documentation

This site is maintained as plain Markdown in [`docs/`](https://github.com/jhnrckmnznrs/fast-ballmapper/tree/main/docs), using GitHub Pages and the built-in Jekyll renderer.

To publish it **without adding another Git branch**, choose **Settings → Pages → Deploy from a branch → main → /docs → Save**. GitHub Pages will build the site from updates to `main/docs`.

Edit the page Markdown, and use [`docs/assets/site.css`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/docs/assets/site.css) for presentation. No JavaScript framework, remote fonts, or separate deployment branch is required.

## Citation

Software citation metadata are in [`CITATION.cff`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/CITATION.cff). The project is distributed under the [MIT license](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/LICENSE).
