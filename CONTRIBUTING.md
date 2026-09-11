# Contributing

Contributions are welcome, especially new range-query backends, correctness
checks, benchmarks, and documentation improvements.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plot]"
```

To exercise the optional CPU approximate backends as well:

```bash
python -m pip install -e ".[dev,plot,faiss,hnswlib]"
```

## Checks

Before opening a pull request, run:

```bash
ruff check .
pytest --cov=fast_ballmapper --cov-report=term-missing
python -m build
python -m twine check dist/*
```

New exact backends should be checked against `BruteForceBackend` on boundary
cases as well as ordinary random inputs. New approximate backends should make
candidate limits and representation precision explicit in `BackendMetadata`.

## Numerical convention

Ball Mapper neighborhoods are closed balls, `d(x, l) <= eps`. Backends with
strict threshold APIs must use the centralized helpers in `_radius.py`; do not
introduce backend-specific tolerances or pre-expand `eps` in callers.
