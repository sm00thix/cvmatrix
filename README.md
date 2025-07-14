# CVMatrix

[![PyPI Version](https://img.shields.io/pypi/v/cvmatrix.svg)](https://pypi.python.org/pypi/cvmatrix/)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/cvmatrix)](https://pypi.python.org/pypi/cvmatrix/)

[![Python Versions](https://img.shields.io/pypi/pyversions/cvmatrix.svg)](https://pypi.python.org/pypi/cvmatrix/)

[![License](https://img.shields.io/pypi/l/cvmatrix.svg)](https://pypi.python.org/pypi/cvmatrix/)

[![Documentation Status](https://readthedocs.org/projects/cvmatrix/badge/?version=latest)](https://cvmatrix.readthedocs.io/en/latest/?badge=latest)

[![Tests Status](https://github.com/Sm00thix/CVMatrix/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/Sm00thix/CVMatrix/actions/workflows/test_workflow.yml)

[![Package Status](https://github.com/Sm00thix/CVMatrix/actions/workflows/package_workflow.yml/badge.svg)](https://github.com/Sm00thix/CVMatrix/actions/workflows/package_workflow.yml)

The [`cvmatrix`](https://pypi.org/project/cvmatrix/) package implements the fast cross-validation algorithms by Engstrøm and Jensen [[1]](#references) for computation of training set $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$ in a cross-validation setting. In addition to correctly handling arbitrary row-wise pre-processing, the algorithms allow for and efficiently and correctly handle any combination of column-wise centering and scaling of `X` and `Y` based on training set statistical moments.

For an implementation of the fast cross-validation algorithms combined with Improved Kernel Partial Least Squares [[2]](#references), see the Python package [`ikpls`](https://pypi.org/project/ikpls/) by Engstrøm et al. [[3]](#references).

## NEW IN 2.0.0: Weighted CVMatrix
The `cvmatrix` software package now also features **weigthed matrix produts** $\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}$ **without increasing time or space complexity compared to the unweighted case**. This is due to a generalization of the algorithms by Engstrøm and Jensen [[1]](#references). A new article formally describing the generalization is to be announced.

## Installation

- Install the package for Python3 using the following command:
    ```shell
    pip3 install cvmatrix
    ```

- Now you can import the class implementing all the algorithms with:
    ```python
    from cvmatrix.cvmatrix import CVMatrix
    ```

## Quick Start

### Use the cvmatrix package for fast computation of training set kernel matrices

> ```python
> import numpy as np
> from cvmatrix.cvmatrix import CVMatrix
> from cvmatrix.partitioner import Partitioner
>
> N = 100  # Number of samples.
> K = 50  # Number of features.
> M = 10  # Number of targets.
>
> X = np.random.uniform(size=(N, K)) # Random X data
> Y = np.random.uniform(size=(N, M)) # Random Y data
> folds = np.arange(100) % 5 # 5-fold cross-validation
>
> # Weights must be non-negative. If centering or scaling is used, the sum of weights
> # for any training partition must be greater than zero. If scaling is used, the
> # number of non-negative weights for any training partition must be greater than
> # the ddof provided in the constructor.
> weights = np.random.uniform(size=(N,)) + 0.1
>
> # Instantiate CVMatrix
> cvm = CVMatrix(
>     center_X=True, # Cemter around the weighted mean of X.
>     center_Y=True, # Cemter around the weighted mean of Y.
>     scale_X=True, # Scale by the weighted standard deviation of X.
>     scale_Y=True, # Scale by the weighted standard deviation of Y.
> )
> # Fit on X, Y, and weights
> cvm.fit(X=X, Y=Y, weights=weights)
>
> # Instantiate Partitioner
> p = Partitioner(folds=folds)
>
> # Compute training set XTWX and/or XTWY for each fold
> for fold in p.folds_dict:
>     val_indices = p.get_validation_indices(fold)
>     # Get both XTWX, XTWY, and weighted statistics
>     result = cvm.training_XTX_XTY(val_indices)
>     (training_XTWX, training_XTWY) = result[0]
>     (training_X_mean, training_X_std, training_Y_mean, training_Y_std) = result[1]
>     
>     # Get only XTWX and weighted statistics for X.
>     # Weighted statistics for Y are returned as None as they are not computed when
>     # only XTWX is requested.
>     result = cvm.training_XTX(val_indices)
>     training_XTWX = result[0]
>     (training_X_mean, training_X_std, training_Y_mean, training_Y_std) = result[1]
>     
>     # Get only XTWY and weighted statistics
>     result = cvm.training_XTY(val_indices)
>     training_XTWY = result[0]
>     (training_X_mean, training_X_std, training_Y_mean, training_Y_std) = result[1]

### Examples
In [examples](https://github.com/Sm00thix/CVMatrix/tree/main/examples), you will find:

- [Compute training matrices with CVMatrix](https://github.com/Sm00thix/CVMatrix/tree/main/examples/training_matrices.py)

## Benchmarks

In [benchmarks](https://github.com/Sm00thix/CVMatrix/tree/main/benchmarks), we have benchmarked cross-validation of the fast algorithms in [`cvmatrix`](https://pypi.org/project/cvmatrix/) against the baseline algorithms implemented in [NaiveCVMatrix](https://github.com/Sm00thix/CVMatrix/tree/main/tests/naive_cvmatrix.py).

<p align=center>
   <img src="https://github.com/Sm00thix/CVMatrix/blob/main/benchmarks/benchmark_cvmatrix_vs_naive.png" width="400" height="400" /> <img src="https://github.com/Sm00thix/CVMatrix/blob/main/benchmarks/benchmark_cvmatrix.png" width="400" height="400"/>
   <br>
   <em> <strong>Left:</strong> Benchmarking cross-validation with the CVMatrix implementation versus the baseline implementation using three common combinations of (column-wise) centering and scaling. <strong>Right:</strong> Benchmarking cross-validation with the CVMatrix implementation for all possible combinations of (column-wise) centering and scaling. Here, most of the graphs lie on top of eachother. In general, no preprocessing is faster than centering which, in turn, is faster than scaling. </em>
</p>

## Contribute

To contribute, please read the [Contribution
Guidelines](https://github.com/Sm00thix/CVMatrix/blob/main/CONTRIBUTING.md).

## References

1. [Engstrøm, O.-C. G. and Jensen, M. H. (2025). Fast partition-based cross-validation with centering and scaling for $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$. *Journal of Chemometrics*, 39(3).](https://doi.org/10.1002/cem.70008)
2. [Dayal, B. S. and MacGregor, J. F. (1997). Improved PLS algorithms. *Journal of Chemometrics*, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23?)
3. [Engstrøm, O.-C. G. and Dreier, E. S. and Jespersen, B. M. and Pedersen, K. S. (2024). IKPLS: Improved Kernel Partial Least Squares and Fast Cross-Validation Algorithms for Python with CPU and GPU Implementations Using NumPy and JAX. *Journal of Open Source Software*, 9(99).](https://doi.org/10.21105/joss.06533)

## Funding
- Up until May 31st 2025, this work has been carried out as part of an industrial Ph. D. project receiving funding from [FOSS Analytical A/S](https://www.fossanalytics.com/) and [The Innovation Fund Denmark](https://innovationsfonden.dk/en). Grant number 1044-00108B.
- From June 1st 2025 and onward, this work is sponsored by [FOSS Analytical A/S](https://www.fossanalytics.com/).
