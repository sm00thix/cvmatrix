# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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