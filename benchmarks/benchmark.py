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

from itertools import product
from time import perf_counter
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
    csv_path = os.environ.get("BENCH_CSV", "benchmark_results.csv")
    try:
        with open(csv_path, "x") as f:
            f.write(
                "model,weights,P,N,K,M,center_X,center_Y,scale_X,scale_Y,time,version\n"
            )
    except FileExistsError:
        pass
    with open(csv_path, "a") as f:
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
    backend: str = "numpy",
    batch_size: int = 10000,
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

    # Create the model. The CVMatrix model takes a `backend` ("numpy" or "jax").
    if model_class is CVMatrix:
        model = CVMatrix(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            dtype=X.dtype,
            copy=True,
            backend=backend,
        )
    else:
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
    elif backend == "jax":
        # Batch the per-fold training matrices over folds with jax.vmap (grouping by
        # validation-set size so shapes are fixed, and chunking by `batch_size` to bound
        # memory).
        import jax
        import jax.numpy as jnp
        from collections import defaultdict

        buckets: dict[int, list] = defaultdict(list)
        for fold in p.folds_dict:
            vi = p.get_validation_indices(fold)
            buckets[vi.shape[0]].append(vi)
        vmapped = jax.jit(jax.vmap(model.training_XTX_XTY))
        for vis in buckets.values():
            stack = jnp.asarray(np.stack(vis))
            for s in range(0, len(vis), batch_size):
                jax.block_until_ready(vmapped(stack[s : s + batch_size]))
    else:
        # Compute the training set matrices
        for fold in p.folds_dict:
            # Get the validation indices for the current fold
            validation_indices = p.get_validation_indices(fold)
            model.training_XTX_XTY(validation_indices)


def benchmark_jax_variants(
    cv_splits, center_X, center_Y, scale_X, scale_Y, X, Y, weights, batch_size
):
    """
    Times the per-fold training-matrix computation (vmapped over folds, chunked by
    `batch_size`) under three JAX execution modes, each including the one-time fit so the
    result is the total cross-validation time (comparable to the numpy/naive totals):

    - "nojit": eager ``jax.vmap`` (no JIT compilation).
    - "coldjit": ``jax.jit(jax.vmap(...))`` timed on its first call (compilation + run).
    - "warmjit": the same jitted function timed on a subsequent call (run only).

    Returns a dict mapping each mode to its total time in seconds.
    """
    import jax
    import jax.numpy as jnp
    from collections import defaultdict

    p = Partitioner(folds=cv_splits)

    t0 = perf_counter()
    model = CVMatrix(
        center_X=center_X, center_Y=center_Y, scale_X=scale_X, scale_Y=scale_Y,
        dtype=X.dtype, copy=True, backend="jax",
    )
    model.fit(X, Y, weights)
    # Stack the per-fold validation indices into fixed-shape batches (one per fold size).
    buckets = defaultdict(list)
    for fold in p.folds_dict:
        vi = p.get_validation_indices(fold)
        buckets[vi.shape[0]].append(vi)
    batches = [jnp.asarray(np.stack(vis)) for vis in buckets.values()]
    fit_time = perf_counter() - t0

    def run(vf):
        for stack in batches:
            for s in range(0, stack.shape[0], batch_size):
                jax.block_until_ready(vf(stack[s : s + batch_size]))

    # No JIT: eager vmap.
    vf_nojit = jax.vmap(model.training_XTX_XTY)
    t0 = perf_counter()
    run(vf_nojit)
    nojit = perf_counter() - t0

    # JIT: the first call compiles (cold); a subsequent call reuses it (warm).
    vf_jit = jax.jit(jax.vmap(model.training_XTX_XTY))
    t0 = perf_counter()
    run(vf_jit)
    coldjit = perf_counter() - t0
    t0 = perf_counter()
    run(vf_jit)
    warmjit = perf_counter() - t0

    return {
        "nojit": fit_time + nojit,
        "coldjit": fit_time + coldjit,
        "warmjit": fit_time + warmjit,
    }


if __name__ == "__main__":
    seed = 42  # Seed for reproducibility
    rng = np.random.default_rng(seed=seed)
    N = int(os.environ.get("BENCH_N", 100000))  # 100k samples
    K = int(os.environ.get("BENCH_K", 500))  # 500 features
    M = int(os.environ.get("BENCH_M", 10))  # 10 targets
    dtype = np.float64  # Data type
    X = rng.random((N, K), dtype=dtype)  # Random X matrix
    Y = rng.random((N, M), dtype=dtype)  # Random Y matrix
    weights = rng.random((N,), dtype=dtype)  # Random weights
    cv_splits = np.arange(N)  # We can use mod P for P-fold cross-validation
    use_weights = [True, False]  # Whether to use weights or not
    # Preprocessing combinations. BENCH_CONFIGS="plot" restricts to the three
    # combinations shown in benchmark_cvmatrix_vs_naive.png (and the ones for which
    # NaiveCVMatrix is benchmarked): no preprocessing, centering only, and centering +
    # scaling. "all" (the default) sweeps every combination of the four flags.
    if os.environ.get("BENCH_CONFIGS", "all") == "plot":
        preprocessing_configs = [
            (False, False, False, False),
            (True, True, False, False),
            (True, True, True, True),
        ]
    else:
        preprocessing_configs = list(
            product([True, False], [True, False], [True, False], [True, False])
        )
    # Ps = [3, 5, 10, 100, 1000, 10000, 100000]
    Ps = [int(p) for p in os.environ.get("BENCH_PS", "100000").split(",")]
    # CVMatrix backends to benchmark (comma-separated). "jax" requires cvmatrix[jax].
    backends = os.environ.get("BENCH_BACKENDS", "numpy").split(",")
    batch_size = int(os.environ.get("BENCH_BATCH", 10000))

    # BENCH_JAX_VARIANTS=1: instead of the numpy/naive/jax sweep, benchmark the JAX
    # backend under no-JIT / cold-JIT / warm-JIT (weighted only) for the figure
    # benchmark_jax_variants.png.
    if os.environ.get("BENCH_JAX_VARIANTS", "0") == "1":
        import jax

        platform = jax.devices()[0].platform
        for (center_X, center_Y, scale_X, scale_Y), P in product(
            preprocessing_configs, Ps
        ):
            print(
                f"JAX variants ({platform}): P={P}, center_X={center_X}, "
                f"center_Y={center_Y}, scale_X={scale_X}, scale_Y={scale_Y}"
            )
            mode_times = benchmark_jax_variants(
                cv_splits % P, center_X, center_Y, scale_X, scale_Y,
                X, Y, weights, batch_size,
            )
            for mode, t in mode_times.items():
                print(f"  CVMatrix-jax-{platform}-{mode}: {t:.2f} seconds")
                save_result_to_csv(
                    f"CVMatrix-jax-{platform}-{mode}",
                    True,
                    P, N, K, M,
                    center_X, center_Y, scale_X, scale_Y,
                    t, __version__,
                )
        sys.exit(0)

    for use_w, (center_X, center_Y, scale_X, scale_Y), P in product(
        use_weights, preprocessing_configs, Ps
    ):
        print(
            f"weights={use_w}, "
            f"P={P}, "
            f"center_X={center_X}, center_Y={center_Y}, "
            f"scale_X={scale_X}, scale_Y={scale_Y}, "
        )
        for backend in backends:
            time = timeit(
                stmt=lambda backend=backend: execute_algorithm(
                    model_class=CVMatrix,
                    cv_splits=cv_splits % P,
                    center_X=center_X,
                    center_Y=center_Y,
                    scale_X=scale_X,
                    scale_Y=scale_Y,
                    X=X,
                    Y=Y,
                    weights=weights if use_w else None,
                    backend=backend,
                    batch_size=batch_size,
                ),
                number=1,
            )
            if backend == "jax":
                import jax

                # Distinguish CPU vs GPU/TPU JAX runs in the results.
                model_name = f"CVMatrix-jax-{jax.devices()[0].platform}"
            else:
                model_name = (
                    "CVMatrix" if backend == "numpy" else f"CVMatrix-{backend}"
                )
            print(f"{model_name}, Time: {time:.2f} seconds")
            save_result_to_csv(
                model_name,
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

        # NaiveCVMatrix is only benchmarked for the three preprocessing combinations
        # shown in benchmark_cvmatrix_vs_naive.png (no preprocessing, centering only, and
        # centering + scaling). It is extremely slow (hours at large P), so it only runs
        # when BENCH_NAIVE=1 is set explicitly.
        if os.environ.get("BENCH_NAIVE", "0") == "1" and (
            center_X == center_Y == scale_X == scale_Y
            or center_X == center_Y is True
            and scale_X == scale_Y is False
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
                    weights=weights if use_w else None,
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
