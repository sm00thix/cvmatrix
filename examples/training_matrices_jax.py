r"""
This file demonstrates the optional **JAX backend** of CVMatrix (``backend="jax"``).
With the JAX backend, the per-fold training matrices
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` (with training-set centering/scaling
and weighting) are computed with ``jax.numpy``. Because each ``training_*`` call is a
pure function of the (closed-over) dataset-wide statistics and the validation indices,
it can be traced by ``jax.jit`` and -- most usefully -- batched over folds with
``jax.vmap`` to run all folds together on a CPU/GPU/TPU.

The numerical results are identical to the default ``backend="numpy"``.

Note: requires the optional JAX dependency, installed with ``pip install cvmatrix[jax]``.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import jax
import jax.numpy as jnp
import numpy as np

from cvmatrix import CVMatrix, Partitioner

# Honor float64 (JAX defaults to float32). CVMatrix also enables this automatically when
# constructed with backend="jax" and a 64-bit dtype.
jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, K, M = 100, 10, 3
    X = rng.uniform(size=(N, K))
    Y = rng.uniform(size=(N, M))
    weights = rng.uniform(size=(N,)) + 0.1  # non-negative

    # 5-fold cross-validation (equal-size folds -> a single vmap shape).
    folds = np.arange(N) % 5

    # Build a JAX-backed CVMatrix. The only change from the numpy backend is
    # backend="jax"; the API is identical.
    cvm = CVMatrix(
        center_X=True, center_Y=True, scale_X=True, scale_Y=True, backend="jax"
    )
    cvm.fit(X, Y, weights)

    # Partitioner does host-side fold/index bookkeeping (numpy). Stack the per-fold
    # validation indices into a single (n_folds, fold_size) array and move it to the
    # device once; jax.vmap then computes every fold's training matrices together.
    p = Partitioner(folds)
    val_index_batch = jnp.asarray(
        np.stack([p.get_validation_indices(fold) for fold in p.folds_dict])
    )

    # vmap (optionally under jit) the per-fold training-matrix computation over folds.
    batched_training_XTX_XTY = jax.jit(jax.vmap(cvm.training_XTX_XTY))
    (XTWX, XTWY), (X_mean, X_std, Y_mean, Y_std) = batched_training_XTX_XTY(
        val_index_batch
    )

    # The leading axis of each output indexes the folds.
    print(f"Folds: {list(p.folds_dict.keys())}")
    print(f"Batched training XTWX shape: {XTWX.shape}  (n_folds, K, K)")
    print(f"Batched training XTWY shape: {XTWY.shape}  (n_folds, K, M)")
    print(f"Batched training X mean shape: {X_mean.shape}  (n_folds, 1, K)")

    # Cross-check the first fold against the numpy backend (identical results).
    cvm_np = CVMatrix(
        center_X=True, center_Y=True, scale_X=True, scale_Y=True, backend="numpy"
    )
    cvm_np.fit(X, Y, weights)
    first_fold = next(iter(p.folds_dict))
    (XTWX_np, XTWY_np), _ = cvm_np.training_XTX_XTY(p.get_validation_indices(first_fold))
    print(
        "Max |jax - numpy| for fold "
        f"{first_fold!r}: XTWX={np.max(np.abs(np.asarray(XTWX[0]) - XTWX_np)):.2e}, "
        f"XTWY={np.max(np.abs(np.asarray(XTWY[0]) - XTWY_np)):.2e}"
    )
