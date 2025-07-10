"""
This script benchmarks the computation of the training set matrices using both the fast
and the naive algorithms as described in the article by Engstrøm. The algorithms are
compared for different values of P, center_X, center_Y, scale_X, and scale_Y. The
results are saved to a CSV file for further analysis.

O.-C. G. Engstrøm and M. H. Jensen (2025):
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import os
import sys

# Add the parent directory of 'CVMatrix' to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the number of threads to 1. This must be done before importing numpy.
os.environ["OMP_NUM_THREADS"] = "1"

from itertools import product
from timeit import timeit
from typing import Hashable, Iterable, Union

import numpy as np

from cvmatrix.__init__ import __version__
from cvmatrix.cvmatrix import CVMatrix
from cvmatrix.partitioner import Partitioner
from tests.naive_cvmatrix import NaiveCVMatrix


def save_result_to_csv(
    model, use_weights, P, N, K, M, center_X, center_Y, scale_X, scale_Y, time, version
):
    try:
        with open("benchmark_results.csv", "x") as f:
            f.write(
                "model,weights,P,N,K,M,"
                "center_X,center_Y,scale_X,scale_Y,time,version\n"
            )
    except FileExistsError:
        pass
    with open("benchmark_results.csv", "a") as f:
        f.write(
            f"{model},{use_weights},{P},{N},{K},{M},"
            f"{center_X},{center_Y},{scale_X},{scale_Y},"
            f"{time},{version}\n"
        )


def execute_algorithm(
    model_class: Union[NaiveCVMatrix, CVMatrix],
    cv_splits: Iterable[Hashable],
    center_X: bool,
    center_Y: bool,
    scale_X: bool,
    scale_Y: bool,
    X: np.ndarray,
    Y: np.ndarray,
    weights: Union[None, np.ndarray],
):
    """
    Execute the computation of the training set matrices
    :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
    and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` using the naive algorithms
    implemented in NaiveCVMatrix and the fast algorithms implemented in CVMatrix.

    Parameters
    ----------
    model_class : Union[NaiveCVMatrix, CVMatrix]
        The model class to use for the computation of the training set matrices.

    cv_splits : Iterable[Hashable]
        The cross-validation splits.

    center_X : bool
        Whether to center `X`.

    center_Y : bool
        Whether to center `Y`.

    scale_X : bool
        Whether to scale `X`.

    scale_Y : bool
        Whether to scale `Y`.

    X : np.ndarray
        The input matrix with shape (N, K).

    Y : np.ndarray
        The target matrix with shape (N, M).

    weights : Union[None, np.ndarray]
        The weights for the samples, if any. If None, no weights are used.
    """

    # Create the model
    model = model_class(
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        dtype=X.dtype,
        copy=True,
    )

    # Create the validation partitioner
    p = Partitioner(folds=cv_splits)

    # Fit the model
    model.fit(X, Y, weights)

    if isinstance(model, NaiveCVMatrix):
        # Compute the training set matrices
        for fold in p.folds_dict:
            # Get the training indices for the current fold
            training_indices = np.concatenate(
                [p.get_validation_indices(f) for f in p.folds_dict if f != fold]
            )
            model.training_XTX_XTY(training_indices)
    else:
        # Compute the training set matrices
        for fold in p.folds_dict:
            # Get the validation indices for the current fold
            validation_indices = p.get_validation_indices(fold)
            model.training_XTX_XTY(validation_indices)


if __name__ == "__main__":
    seed = 42  # Seed for reproducibility
    rng = np.random.default_rng(seed=seed)
    N = 100000  # 100k samples
    K = 500  # 500 features
    M = 10  # 10 targets
    dtype = np.float64  # Data type
    X = rng.random((N, K), dtype=dtype)  # Random X matrix
    Y = rng.random((N, M), dtype=dtype)  # Random Y matrix
    weights = rng.random((N,), dtype=dtype)  # Random weights
    cv_splits = np.arange(N)  # We can use mod P for P-fold cross-validation
    use_weights = [True, False]  # Whether to use weights or not
    center_Xs = [True, False]
    center_Ys = [True, False]
    scale_Xs = [True, False]
    scale_Ys = [True, False]
    Ps = [3, 5, 10, 100, 1000, 10000, 100000]

    for use_w, center_X, center_Y, scale_X, scale_Y, P in product(
        use_weights, center_Xs, center_Ys, scale_Xs, scale_Ys, Ps
    ):
        print(
            f"weights={use_w}, "
            f"P={P}, "
            f"center_X={center_X}, center_Y={center_Y}, "
            f"scale_X={scale_X}, scale_Y={scale_Y}, "
        )
        time = timeit(
            stmt=lambda: execute_algorithm(
                model_class=CVMatrix,
                cv_splits=cv_splits % P,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                X=X,
                Y=Y,
                weights=weights if use_weights else None,
            ),
            number=1,
        )
        print(f"CVMatrix, Time: {time:.2f} seconds")
        save_result_to_csv(
            "CVMatrix",
            use_w,
            P,
            N,
            K,
            M,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            time,
            __version__,
        )

        if (
            center_X == center_Y == scale_X == scale_Y
            or center_X == center_Y == True
            and scale_X == scale_Y == False
        ):
            time = timeit(
                stmt=lambda: execute_algorithm(
                    model_class=NaiveCVMatrix,
                    cv_splits=cv_splits % P,
                    center_X=center_X,
                    center_Y=center_Y,
                    scale_X=scale_X,
                    scale_Y=scale_Y,
                    X=X,
                    Y=Y,
                    weights=weights,
                ),
                number=1,
            )
            print(f"NaiveCVMatrix, Time: {time:.2f} seconds")
            print()
            save_result_to_csv(
                "NaiveCVMatrix",
                use_w,
                P,
                N,
                K,
                M,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                time,
                __version__,
            )
