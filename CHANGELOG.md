# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2026-10-31

### Added
- With `backend="jax"`, `fit` accepts sample weights traced by `jax.jit` or `jax.vmap`, e.g., one weight vector per dataset when vmapping over datasets, as in distance-weighted local PLS. Previously, traced weights raised `TracerBoolConversionError`. As for traced validation indices, data-dependent checks are skipped for traced values: the caller must check the concrete weights before tracing (non-negative; at least one non-zero training weight in each fold when centering or scaling, and more than `ddof` when scaling), as invalid traced weights silently give wrong or NaN results.
- Concrete weights and validation indices are now checked as in eager execution even inside `jax.jit`, e.g., when they are closed over, or when a model fitted eagerly is used in a jitted function. The checks are evaluated at trace time under `jax.ensure_compile_time_eval()`. Previously, with weights, such calls raised `TracerBoolConversionError` even for valid folds.
- `CVMatrix.validate_folds(folds, N, weights=None)`: a host-side pre-flight check for folds and weights that will be traced, where the training methods cannot check them. It performs, on the concrete values, the checks of the training methods (non-negative weights after the cast to `dtype`; non-zero and more than `ddof` non-zero training weights per fold, depending on centering and scaling), and also checks that the validation indices are in the range 0 to `N` - 1, as JAX silently clamps out-of-range indices under tracing. It accepts a `Partitioner`, a mapping from folds to validation indices, or an iterable or 2-D array of validation indices, needs no fitted instance, and lists every failing fold in one `ValueError`.

### Changed
- The CI's `setup-uv` action is updated from v7 to v10.3.3 in the build and test actions.

### Fixed
- Without weights, degenerate folds now raise also when the validation indices are traced by `jax.jit` or `jax.vmap`: the training-set size is known while tracing, as array shapes are. Previously, the check was skipped, and too few training samples for `ddof` silently gave infinite standard deviations.
- Without weights, a fold whose training set is empty now raises the same `ValueError` as with weights when training-set statistics are needed (centering or scaling), instead of silently giving NaN means.
- A `fit` that raises because of negative weights now leaves a previously fitted instance unchanged. Previously, it had already replaced `X`, `Y`, and `weights`, so later calls mixed them with the previous matrix products and statistics.

### Notes
- The NumPy backend's results are unchanged; its only behavior change is the empty-training-set fix above. When constructing a `CVMatrix` with a 64-bit `dtype` inside a function traced by `jax.jit` or `jax.vmap`, enable JAX's 64-bit mode before tracing (now documented under `backend`).

## [3.3.0] - 2026-10-30

### Added
- `compute_XTX` and `compute_XTY` constructor parameters (both default `True`, the previous behavior) that control whether `fit` computes the dataset-wide `XTX` (`X^T W X`) and `XTY` (`X^T W Y`). A product that is never used can now be skipped: e.g., `compute_XTX=False` avoids the O(NK²) time and K² memory of `XTX` when only `training_XTY` and/or `training_statistics` are used, which can dominate `fit` for wide `X`. A product that is not computed is `None`, and the training methods that need it raise a `ValueError`. The products are still computed in `fit`, so they are pickled along with a fitted `CVMatrix` and are not recomputed per task when cross-validating in parallel worker processes (e.g., with `joblib`).
- Support for free-threaded Python. The CI test matrices now include free-threaded Python 3.15 (`3.15t`) on Linux, macOS, and Windows, with a check that no dependency re-enables the GIL. The JAX backend is tested on `3.15t` on Linux and macOS only, as jaxlib publishes no free-threaded wheel for Windows. The package declares the `Programming Language :: Python :: Free Threading :: 3 - Stable` classifier. A fitted `CVMatrix` can be shared by multiple threads, as the per-fold training methods only read it.

### Changed
- The CI test workflow for pushes to `main` now also tests Python 3.15, matching the pull-request workflow.
- The per-fold training methods now gather only the validation set rows they use. Previously, the unweighted validation set rows of `X` were gathered, but not used, by `training_XTY` and `training_statistics`. With `weights` on the numpy backend, this makes `training_XTY` ~20% faster and `training_statistics` ~25-45% faster. Results are unchanged. Unweighted computations and computations traced by `jax.jit` were not affected.

### Fixed
- A fitted `CVMatrix` can be pickled with the standard library again. Since 3.2.0, the instance held the array namespace `xp` (the `numpy` or `jax.numpy` module), which cannot be pickled, so `pickle`, `copy.deepcopy`, `multiprocessing`, and `concurrent.futures.ProcessPoolExecutor` failed with `TypeError: cannot pickle 'module' object` (`joblib`'s default `loky` backend was not affected, as it pickles with `cloudpickle`). The module is now left out of the pickled state and resolved again from `backend` when unpickling. With `backend="jax"`, JAX's 64-bit mode is enabled before the arrays are restored, so float64 data stays float64 in fresh worker processes, where it would otherwise silently become float32.
- Re-fitting a `CVMatrix` now replaces all state derived from `Y` instead of keeping it from the previous fit. Re-fitting without `Y` resets `XTY`, `WY`, `sum_Y`, `sq_Y`, and `sum_sq_Y` to `None`, and re-fitting with weights that do not require `WY` (no `center_X`, `center_Y`, or `scale_Y`) resets `WY` to `None`; likewise, `sq_X` is reset to `None` when `scale_X` is `False`. Results were not affected, but the stale arrays were kept in memory and pickled along with the fitted model (e.g., to parallel worker processes).

## [3.2.3] - 2026-10-30
- Increased the maximum supported Python version from 3.14 to 3.15. Python 3.15 is now included in the CI test matrices. No functional changes.

## [3.2.2] - 2026-07-06

### Changed
- Lowered the minimum supported Python version from 3.11 to 3.10 (`requires-python = ">=3.10, <3.15"`). The code base uses no Python-3.11-only features; this broadens compatibility for downstream packages that still support Python 3.10. Python 3.10 is now included in the CI test matrices. No functional changes.

### Fixed
- The fold-index parameters (`validation_indices` / `val_indices`) are now annotated with a new `IndexArray` alias that `_enable_jax_typing()` broadens to admit `jax.Array`, matching the `Array`/`Scalar` aliases introduced in 3.2.1. Previously they were annotated as `npt.NDArray[np.int_]` only, so running the test suite with `typeguard` instrumentation enabled failed on the `jax.jit`/`jax.vmap` trace tests (the indices are abstract tracers under tracing). Runtime behavior is unchanged; this only affects runtime type checking.
- CI now actually enables `typeguard` instrumentation during tests: the test command passed `--typeguard-packages=cvmatrix/` (a path, with a trailing slash) instead of the package name `cvmatrix`, which silently instrumented nothing and masked the annotation issue above.

## [3.2.1] - 2026-07-01

### Changed
- `import cvmatrix` no longer imports JAX, even when JAX is installed. JAX was previously imported at module load time to broaden the array/scalar type hints so that `backend="jax"` values satisfy runtime type checking; it is now imported lazily, only when the JAX backend is actually resolved (`backend="jax"`). The array/scalar type aliases start NumPy-only and are broadened in-place to also admit `jax.Array` the first time the JAX backend is used (annotations are deferred via `from __future__ import annotations`, so `typeguard` resolves them against the broadened aliases at call time). This keeps NumPy-only import paths — and downstream packages that use only the NumPy backend — free of the JAX import cost. The `numpy` and `jax` backends are behavior- and result-identical to 3.2.0.

## [3.2.0] - 2026-06-29

### Added
- Optional **JAX backend** for `CVMatrix` via `backend="jax"` (a `Literal["numpy", "jax"]`; default remains `"numpy"`). All array operations are routed through a resolved array namespace (`numpy` or `jax.numpy`), so the per-fold `training_XTX`/`training_XTY`/`training_XTX_XTY`/`training_statistics` computations can be traced by `jax.jit` and batched with `jax.vmap` on CPU/GPU/TPU. Install with `cvmatrix[jax]`.
- Backend-neutral, trace-safe standard-deviation clamping (`maximum`/`where` instead of boolean-mask assignment). Degenerate-fold validation (`ValueError`s) still fires for eager (numpy or concrete jax) execution; under `jax.jit`/`jax.vmap` tracing the check is deferred to a host-side pre-flight by the caller.

### Changed
- Switched build and development tooling from Poetry to [uv](https://docs.astral.sh/uv/): the project now uses PEP 621 `[project]` metadata, the `hatchling` build backend, and a PEP 735 `[dependency-groups]` dev group (`poetry.lock` is replaced by `uv.lock`). The published package, its runtime dependencies, and the `jax` extra are unchanged.

### Notes
- The numpy backend is byte-identical to previous releases (verified against the existing test suite) and performance-neutral.

## [3.1.6] - 2025
### Changed
- `CVMatrix` and `Partitioner` can now be imported with `from cvmatrix import CVMatrix, Partitioner` or `import cvmatrix` followed by `cvmatrix.CVMatrix` and `cvmatrix.Partitioner`.

## [3.0.0] - 2025

### Added
- `Partitioner` class for managing cross-validation folds

### Changed
- `CVMatrix` no longer internally stores a dictionairy of validation indices. This responsibility is offloaded to `Partitioner` so `CVMatrix` can be pickled more efficiently when used in a multiprocessing context such as by the ikpls package [[1]](#references).

## [2.0.0] - 2024

### Added
- **Weighted matrix products** $\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}$ support without increasing time or space complexity
- Generalization of the fast cross-validation algorithms by Engstrøm and Jensen to handle weighted cases
- Support for weighted centering and weighted scaling based on training set statistical moments
- All 16 (12 unique) combinations of weighted column-wise centering and scaling for X and Y matrices

### Changed
- Extended algorithms to correctly handle weighted cases while maintaining efficiency
- Enhanced `CVMatrix` class to accept sample weights

### Notes
- The weighted extension maintains the same computational complexity as the unweighted algorithms
- Formal description of the weighted generalization to be announced in upcoming publication

## [1.0.0] - Initial Release

### Added
- Implementation of fast cross-validation algorithms by Engstrøm and Jensen [[2]](#references) for computation of training set $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$
- `CVMatrix` class for efficient kernel matrix computation in cross-validation settings
- Support for arbitrary row-wise preprocessing of data
- Support for column-wise centering and scaling of X and Y based on training set statistics
- Methods for computing:
  - `training_XTX_XTY()`: Both training set kernel matrices and statistics
  - `training_XTX()`: Training set $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and X statistics
  - `training_XTY()`: Training set $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$ and statistics
- Comprehensive test suite and benchmarks
- Documentation at cvmatrix.readthedocs.io

### Features
- Correct handling of column-wise centering and scaling without data leakage
- Efficient computation of training set moments (means and standard deviations)
- Support for all combinations of centering and scaling options for X and Y
- Significantly faster than naive implementations for cross-validation scenarios

## References

1. [Engstrøm, O.-C. G. and Jensen, M. H. (2025). Fast Partition-Based Cross-Validation With Centering and Scaling for $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008)
2. [IKPLS. Fast CPU and GPU Python implementations of Improved Kernel Partial Least Squares (PLS) by Dayal and MacGregor (1997) and Fast Partition-Based Cross-Validation With Centering and Scaling for XTX and XTY by Engstrøm and Jensen (2025). This package also includes options to use sample weights for PLS modeling.](https://github.com/sm00thix/ikpls)