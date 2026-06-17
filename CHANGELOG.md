# Changelog

## 0.1.0

- Reorganized the project into a PyPI-ready `src` layout.
- Renamed the distribution to `fast-ballmapper` and the import package to
  `fast_ballmapper`.
- Converted the public API and internal Python identifiers to snake_case.
- Split landmark selection, backends, graph construction, coloring, and plotting
  into focused modules.
- Made FAISS, Matplotlib, and Plotly optional extras.
- Added automated tests and package build metadata.
