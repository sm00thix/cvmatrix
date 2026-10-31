"""
This file contains tests for the cvmatrix package. In general, the tests compare the
output of the fast algorithms implemented in the CVMatrix class with the output of the
naive algorithms implemented in the NaiveCVMatrix class, both of which are described in
the article by Engstrøm. Some of the tests are performed on a real dataset of NIR
spectra and ground truth values for 8 different grain varieties, protein, and moisture.
This dataset is publicly available on GitHub and originates from the articles by Dreier
et al. and Engstrøm et al. See the load_data module for more information about the
dataset.

O.-C. G. Engstrøm and M. H. Jensen (2025):
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import multiprocessing
import pickle
import sys
from collections.abc import Hashable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from typing import Iterable, Union

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

from cvmatrix.cvmatrix import CVMatrix
from cvmatrix.partitioner import Partitioner

from . import load_data
from .naive_cvmatrix import NaiveCVMatrix


class TestClass:
    """
    Class for testing the CVMatrix implementation.

    This class contains methods for testing the CVMatrix class. In particular, this
    class tests for equivalency between the naive, straight-forward algorithms
    implemented in the NaiveCVMatrix class and the fast algorithms implemented in the
    CVMatrix class. The tests are performed on a real dataset of NIR spectra and ground
    truth values for 8 different grain varieties, protein, and moisture. The dataset is
    publicly available on GitHub and originates from the articles by Dreier et al. and
    Engstrøm et al. See the load_data module for more information about the dataset.
    """

    csv = load_data.load_csv()
    raw_spectra = load_data.load_spectra()
    seed = 42
    ones_weights = np.ones(csv.shape[0], dtype=np.float64)
    rng = np.random.default_rng(seed=seed)
    random_weights = rng.random(csv.shape[0]).astype(np.float64)

    # # Randomly shuffle and cutoff the dataset to 500 samples
    # indices = rng.choice(np.arange(csv.shape[0]), size=500, replace=False)
    # csv = csv.iloc[indices].reset_index(drop=True)
    # raw_spectra = raw_spectra[indices, :]
    # ones_weights = ones_weights[indices]
    # random_weights = random_weights[indices]

    @pytest.fixture(params=["numpy", "jax"], autouse=True)
    def _set_backend(self, request):
        """
        Run every test with both the "numpy" and "jax" CVMatrix backends. The jax
        parametrization is skipped if JAX is not installed. `fit_fast` builds the
        CVMatrix model with `backend=self.backend`, so all equivalence and behavior
        tests exercise both backends.
        """
        if request.param == "jax":
            pytest.importorskip("jax")
        self.backend = request.param

    def load_X(self) -> npt.NDArray[np.float64]:
        """
        Loads the raw spectral data.

        Returns
        -------
        npt.NDArray[np.float64]
            A copy of the raw spectral data.
        """
        return np.copy(self.raw_spectra)

    def load_Y(self, names: list[str]) -> npt.NDArray[np.float64]:
        """
        Loads target values based on the specified column names.

        Parameters
        ----------
        names : list[str]
            The names of the columns to load.

        Returns
        -------
        npt.NDArray[np.float64]
            A copy of the target values.
        """
        return self.csv[names].to_numpy()

    def load_weights(self, random: bool) -> npt.NDArray[np.float64]:
        """
        Loads sample weights
        """
        if random:
            return np.copy(self.random_weights)
        return np.copy(self.ones_weights)

    def randomly_zero_weights(
        self, weights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Randomly sets some weights to zero.
        """
        # Choose 10% of the weights to set to zero
        num_zero_weights = int(0.1 * weights.shape[0])
        zero_indices = self.rng.choice(
            np.arange(weights.shape[0]), size=num_zero_weights, replace=False
        )
        weights = np.copy(weights)
        weights[zero_indices] = 0.0
        return weights

    def zero_weights_in_fold(
        self, weights: npt.NDArray[np.float64], fold: Hashable
    ) -> npt.NDArray[np.float64]:
        """
        Sets all weights to zero for a specific fold.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            The weights for the observations.

        fold : Hashable
            The fold for which to set the weights to zero.

        Returns
        -------
        npt.NDArray[np.float64]
            The modified weights with all weights for the specified fold set to zero.
        """
        indices = np.where(self.csv["split"] == fold)[0]
        weights[indices] = 0.0
        return weights

    def subset_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Subsets the data based on the cross-validation indices. The subset chosen
        ensures exactly 10 samples are chosen for each fold.
        """
        return np.concatenate(
            [np.where(indices == i)[0][:10] for i in np.unique(indices)]
        )

    def fit_models(
        self,
        X: npt.ArrayLike,
        Y: Union[None, npt.ArrayLike],
        weights: Union[None, npt.ArrayLike],
        folds: Iterable[Hashable],
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
        ddof: int,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
    ) -> tuple[NaiveCVMatrix, CVMatrix, Partitioner]:
        """
        Fits the NaiveCVMatrix and CVMatrix models.

        Parameters
        ----------
        X : npt.ArrayLike
            The predictor variables.

        Y : Union[None, npt.ArrayLike]
            The response variables.

        weights : Union[None, npt.ArrayLike]
            The weights for the observations. If `None`, all observations are equally
            weighted.

        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.

        center_X : bool
            Whether to center `X`.

        center_Y : bool
            Whether to center `Y`.

        scale_X : bool
            Whether to scale `X`.

        scale_Y : bool
            Whether to scale `Y`.

        ddof : int
            The delta degrees of freedom used in the computation of the standard
            deviations.

        dtype : np.floating, optional, default=np.float64
            The data-type of the arrays used in the computation.

        copy : bool, optional, default=True
            Whether to make a copy of the input arrays. If `False` and the input arrays
            are already NumPy arrays of type `dtype`, then no copy is made. If `False`
            and the input arrays are not NumPy arrays of type `dtype`, then a copy is
            made. If `True` a copy is always made. If no copy is made, then external
            modifications to `X` or `Y` will result in undefined behavior.

        Returns
        -------
        tuple[NaiveCVMatrix, CVMatrix]
            A tuple containing the NaiveCVMatrix and CVMatrix models.
        """
        fast = self.fit_fast(
            X,
            Y,
            weights,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            dtype,
            copy,
        )
        naive = self.fit_naive(
            X,
            Y,
            weights,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            dtype,
            copy,
        )
        p = self.instantiate_p(folds)
        return naive, fast, p

    def fit_fast(
        self,
        X: npt.ArrayLike,
        Y: Union[None, npt.ArrayLike],
        weights: Union[None, npt.ArrayLike],
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
        ddof: int,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
    ) -> CVMatrix:
        """
        Fits the CVMatrix model.

        Parameters
        ----------
        X : npt.ArrayLike
            The predictor variables.

        Y : Union[None, npt.ArrayLike]
            The response variables.

        weights : Union[None, npt.ArrayLike]
            The weights for the observations. If `None`, all observations are equally
            weighted.

        center_X : bool
            Whether to center `X`.

        center_Y : bool
            Whether to center `Y`.

        scale_X : bool
            Whether to scale `X`.

        scale_Y : bool
            Whether to scale `Y`.

        ddof : int
            The delta degrees of freedom used in the computation of the standard
            deviations.

        dtype : np.floating, optional, default=np.float64
            The data-type of the arrays used in the computation.

        copy : bool, optional, default=True
            Whether to make a copy of the input arrays. If `False` and the input arrays
            are already NumPy arrays of type `dtype`, then no copy is made. If `False`
            and the input arrays are not NumPy arrays of type `dtype`, then a copy is
            made. If `True` a copy is always made. If no copy is made, then external
            modifications to `X` or `Y` will result in undefined behavior.
        """
        fast = CVMatrix(
            center_X, center_Y, scale_X, scale_Y, ddof, dtype, copy,
            backend=getattr(self, "backend", "numpy"),
        )
        fast.fit(X, Y, weights)
        return fast

    def fit_naive(
        self,
        X: npt.ArrayLike,
        Y: Union[None, npt.ArrayLike],
        weights: Union[None, npt.ArrayLike],
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
        ddof: int,
        dtype: np.floating = np.float64,
        copy: bool = True,
        fast_weight_computation: bool = True,
    ) -> NaiveCVMatrix:
        """
        Fits the NaiveCVMatrix model.

        Parameters
        ----------
        X : npt.ArrayLike
            The predictor variables.

        Y : Union[None, npt.ArrayLike]
            The response variables.

        weights : Union[None, npt.ArrayLike]
            The weights for the observations. If `None`, all observations are equally
            weighted.

        center_X : bool
            Whether to center `X`.

        center_Y : bool
            Whether to center `Y`.

        scale_X : bool
            Whether to scale `X`.

        scale_Y : bool
            Whether to scale `Y`.

        ddof : int
            The delta degrees of freedom used in the computation of the standard
            deviations.

        dtype : np.floating, optional, default=np.float64
            The data-type of the arrays used in the computation.

        copy : bool, optional, default=True
            Whether to make a copy of the input arrays. If `False` and the input arrays
            are already NumPy arrays of type `dtype`, then no copy is made. If `False`
            and the input arrays are not NumPy arrays of type `dtype`, then a copy is
            made. If `True` a copy is always made. If no copy is made, then external
            modifications to `X` or `Y` will result in undefined behavior.
        """
        naive = NaiveCVMatrix(
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            dtype,
            copy,
            fast_weight_computation,
        )
        naive.fit(X, Y, weights)
        return naive

    def instantiate_p(self, folds: Iterable[Hashable]) -> Partitioner:
        """
        Instantiates the Partitioner.

        Parameters
        ----------
        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.

        Returns
        -------
        Partitioner
            The instantiated Partitioner.
        """
        return Partitioner(folds)

    @staticmethod
    def leaves(result) -> list:
        """
        Flattens the possibly nested tuple returned by a CVMatrix training method into
        a list of its arrays and `None`s.
        """
        if isinstance(result, tuple):
            return [leaf for r in result for leaf in TestClass.leaves(r)]
        return [result]

    def check_equivalent_stats(
        self,
        naive_stats: Union[None, npt.NDArray[np.float64]],
        fast_stats: Union[None, npt.NDArray[np.float64]],
        err_msg: str = "",
        atol=1e-8,
    ) -> None:
        """
        Checks if the statistics computed by the NaiveCVMatrix and CVMatrix models are
        equivalent.

        Parameters
        ----------
        naive_stats : Union[None, npt.NDArray[np.float64]]
            The statistics computed by the NaiveCVMatrix model.

        fast_stats : Union[None, npt.NDArray[np.float64]]
            The statistics computed by the CVMatrix model.

        err_msg : str, optional
            An error message to display if the statistics are not equivalent.
        """
        for naive_st, fast_st in zip(naive_stats, fast_stats):
            if fast_st is None or naive_st is None:
                return
            assert_allclose(fast_st, naive_st, err_msg=err_msg, atol=atol)

    def check_equivalent_matrices(
        self,
        model_1: Union[CVMatrix, NaiveCVMatrix],
        model_2: Union[CVMatrix, NaiveCVMatrix],
        p: Partitioner,
        max_folds: Union[int, None] = None,
    ) -> None:
        """
        Checks if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent.

        Parameters
        ----------
        naive : NaiveCVMatrix
            The NaiveCVMatrix model.

        fast : CVMatrix
            The CVMatrix model.

        p : Partitioner
            The Partitioner containing the folds.

        max_folds : Union[int, None], optional, default=None
            The maximum number of folds to check. If `None`, all folds are checked.
            Defaults to `None`.
        """
        error_msg = (
            f"\n"
            f"fast center_X: {model_2.center_X}, center_Y: {model_2.center_Y}, "
            f"scale_X: {model_2.scale_X}, scale_Y: {model_2.scale_Y}"
            f"\nddof: {model_2.ddof}, dtype: {model_2.dtype}, copy: {model_2.copy}, "
            f"weights: {model_2.weights is not None}, "
            f"\n"
            f"\nnaive center_X: {model_1.center_X}, center_Y: {model_1.center_Y}, "
            f"scale_X: {model_1.scale_X}, scale_Y: {model_1.scale_Y}"
            f"\nddof: {model_1.ddof}, dtype: {model_1.dtype}, copy: {model_1.copy}, "
            f"weights: {model_1.weights is not None}"
            f"\n"
        )
        for i, fold in enumerate(p.folds_dict):
            if max_folds is not None and i == max_folds:
                break
            error_msg = f"Fold: {fold}\n{error_msg}"
            validation_indices = p.get_validation_indices(fold)
            training_indices = np.concatenate(
                [p.folds_dict[k] for k in p.folds_dict if k != fold]
            )
            model_1_indices = (
                training_indices
                if isinstance(model_1, NaiveCVMatrix)
                else validation_indices
            )
            model_2_indices = (
                training_indices
                if isinstance(model_2, NaiveCVMatrix)
                else validation_indices
            )
            if model_1.Y is not None:
                # Check if the matrices are equivalent for the training_XTX_XTY method
                # between the NaiveCVMatrix and CVMatrix models.
                (model_2_XTX, model_2_XTY), model_2_stats = model_2.training_XTX_XTY(
                    model_2_indices
                )
                (model_1_XTX, model_1_XTY), model_1_stats = model_1.training_XTX_XTY(
                    model_1_indices
                )
                self.check_equivalent_stats(
                    model_1_stats, model_2_stats, err_msg=error_msg
                )
                assert_allclose(model_2_XTX, model_1_XTX, err_msg=error_msg, atol=1e-8)
                assert_allclose(model_2_XTY, model_1_XTY, err_msg=error_msg, atol=1e-8)
                # Check if the matrices are equivalent for the training_XTX and
                # training_XTY methods between the NaiveCVMatrix and CVMatrix models.
                # Also check if the matrices are equivalent for the training_XTX,
                # training_XTY, and training_XTX_XTY methods.
                direct_model_1_XTX, direct_model_1_XTX_stats = model_1.training_XTX(
                    model_1_indices
                )
                direct_model_2_XTX, direct_model_2_XTX_stats = model_2.training_XTX(
                    model_2_indices
                )
                direct_model_1_XTY, direct_model_1_XTY_stats = model_1.training_XTY(
                    model_1_indices
                )
                direct_model_2_XTY, direct_model_2_XTY_stats = model_2.training_XTY(
                    model_2_indices
                )
                assert_allclose(
                    direct_model_2_XTX, direct_model_1_XTX, err_msg=error_msg, atol=1e-8
                )
                assert_allclose(
                    direct_model_2_XTY, direct_model_1_XTY, err_msg=error_msg, atol=1e-8
                )
                assert_allclose(
                    direct_model_2_XTX, model_2_XTX, err_msg=error_msg, atol=1e-8
                )
                assert_allclose(
                    direct_model_2_XTY, model_2_XTY, err_msg=error_msg, atol=1e-8
                )
                self.check_equivalent_stats(
                    direct_model_1_XTX_stats,
                    direct_model_2_XTX_stats,
                    err_msg=error_msg,
                )
                self.check_equivalent_stats(
                    direct_model_1_XTY_stats,
                    direct_model_2_XTY_stats,
                    err_msg=error_msg,
                )
            else:
                # Check if the matrices are equivalent for the training_XTX method
                # between the NaiveCVMatrix and CVMatrix models.
                model_1_XTX, model_1_XTX_stats = model_1.training_XTX(model_1_indices)
                model_2_XTX, model_2_XTX_stats = model_2.training_XTX(model_2_indices)
                assert_allclose(model_2_XTX, model_1_XTX, err_msg=error_msg, atol=1e-8)
                self.check_equivalent_stats(
                    model_1_XTX_stats, model_2_XTX_stats, err_msg=error_msg
                )

    def test_all_preprocessing_combinations(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent for basic settings.
        """
        X = self.load_X()[:, :5]  # Use only the first 5 variables for faster testing.
        Ys = [None, self.load_Y(["Protein", "Moisture"])]
        weights = self.load_weights(random=True)
        folds = self.load_Y(["split"]).squeeze()
        assert X.shape[0] == Ys[1].shape[0] == folds.shape[0] == weights.shape[0]
        assert len(np.unique(folds)) == 3
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        use_weights = [False, True]
        ddofs = [0, 1]
        for center_X, center_Y, scale_X, scale_Y, use_w, ddof, Y in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys, use_weights, ddofs, Ys
        ):
            if use_w:
                cur_weights = self.randomly_zero_weights(weights)
            else:
                cur_weights = None
            naive, fast, p = self.fit_models(
                X,
                Y,
                cur_weights,
                folds,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
            self.check_equivalent_matrices(naive, fast, p)

    def test_jax_backend_requires_jax(self, monkeypatch):
        """
        Tests that requesting backend="jax" raises ImportError with an install hint when
        JAX is not installed. JAX absence is simulated by hiding it from `sys.modules`,
        so this test is valid regardless of whether JAX is installed in the environment.
        """
        monkeypatch.setitem(sys.modules, "jax", None)
        monkeypatch.setitem(sys.modules, "jax.numpy", None)
        with pytest.raises(ImportError, match="cvmatrix\\[jax\\]"):
            CVMatrix(backend="jax")

    def test_invalid_backend(self):
        """
        Tests that an unknown backend string raises ValueError (the sibling of the
        ImportError path tested in `test_jax_backend_requires_jax`). Exercised via
        `_resolve_backend` directly: the constructor's `Literal["numpy", "jax"]` hint is
        intercepted by typeguard before the ValueError is reached, whereas
        `_resolve_backend(backend: str)` accepts the str and reaches the real check.
        """
        from cvmatrix.cvmatrix import _resolve_backend

        with pytest.raises(ValueError, match="Invalid backend"):
            _resolve_backend("torch")

    def test_jax_jit_vmap_matches_numpy(self):
        """
        Exercises the JAX backend under jax.jit + jax.vmap (the headline feature and the
        trace-safety machinery: the deferred degenerate-fold checks and the xp.maximum/
        xp.where standard-deviation clamp all run under tracing here). Batches the per-
        fold training_XTX_XTY over equal-size folds with jax.vmap and checks the results
        match the eager numpy backend, for both unweighted and weighted folds. The
        unweighted case specifically covers the path where the training-set count is a
        static int rather than a tracer.
        """
        if getattr(self, "backend", "numpy") != "jax":
            pytest.skip("JAX-specific trace test; runs under the jax parametrization.")
        pytest.importorskip("jax")
        import jax
        import jax.numpy as jnp

        X = self.load_X()[:, :20]
        Y = self.load_Y(["Protein", "Moisture"])
        # Equal-size folds so jax.vmap sees a fixed (n_folds, fold_size) index batch.
        n_folds = 5
        n = (X.shape[0] // n_folds) * n_folds
        X, Y = X[:n], Y[:n]
        folds = np.arange(n) % n_folds
        p = Partitioner(folds)
        val_index_batch = jnp.asarray(
            np.stack([p.get_validation_indices(f) for f in p.folds_dict])
        )

        for weights in (None, self.load_weights(random=True)[:n]):
            cvm_jax = CVMatrix(
                center_X=True, center_Y=True, scale_X=True, scale_Y=True,
                ddof=1, backend="jax",
            )
            cvm_jax.fit(X, Y, weights)
            cvm_np = CVMatrix(
                center_X=True, center_Y=True, scale_X=True, scale_Y=True,
                ddof=1, backend="numpy",
            )
            cvm_np.fit(X, Y, weights)
            (XTWX_b, XTWY_b), (Xm_b, Xs_b, Ym_b, Ys_b) = jax.jit(
                jax.vmap(cvm_jax.training_XTX_XTY)
            )(val_index_batch)
            for i, fold in enumerate(p.folds_dict):
                (XTWX, XTWY), (Xm, Xs, Ym, Ys) = cvm_np.training_XTX_XTY(
                    p.get_validation_indices(fold)
                )
                assert_allclose(np.asarray(XTWX_b[i]), XTWX, atol=1e-8, rtol=1e-8)
                assert_allclose(np.asarray(XTWY_b[i]), XTWY, atol=1e-8, rtol=1e-8)
                assert_allclose(np.asarray(Xm_b[i]), Xm, atol=1e-8, rtol=1e-8)
                assert_allclose(np.asarray(Xs_b[i]), Xs, atol=1e-8, rtol=1e-8)
                assert_allclose(np.asarray(Ym_b[i]), Ym, atol=1e-8, rtol=1e-8)
                assert_allclose(np.asarray(Ys_b[i]), Ys, atol=1e-8, rtol=1e-8)

    def test_jax_trace_degenerate_folds(self):
        """
        Tests the degenerate-fold checks with validation indices traced by
        jax.jit/jax.vmap. Without weights, the training-set size is static while
        tracing, so degenerate folds raise as in eager execution: an empty training
        set, and one with no more samples than `ddof`. With weights, the fold's count
        of non-zero training weights depends on the traced indices, so the checks are
        deferred to a host-side pre-flight and tracing must complete without raising.
        """
        if getattr(self, "backend", "numpy") != "jax":
            pytest.skip("JAX-specific trace test; runs under the jax parametrization.")
        pytest.importorskip("jax")
        import jax
        import jax.numpy as jnp

        X = self.load_X()[:6, :4]
        Y = self.load_Y(["Protein"])[:6]
        # Validating on all 6 rows leaves no training samples; validating on 5 leaves 1,
        # which is not more than ddof=1.
        for degenerate_val, match in (
            (np.arange(6), "greater than zero"),
            (np.arange(5), "greater than `ddof`"),
        ):
            cvm = CVMatrix(ddof=1, backend="jax")
            cvm.fit(X, Y, None)  # unweighted
            with pytest.raises(ValueError, match=match):
                cvm.training_XTX_XTY(degenerate_val)
            with pytest.raises(ValueError, match=match):
                jax.jit(cvm.training_XTX_XTY)(jnp.asarray(degenerate_val))
            with pytest.raises(ValueError, match=match):
                jax.vmap(cvm.training_XTX_XTY)(jnp.asarray(degenerate_val)[None])

        weights = np.ones(6)
        weights[5] = 0.0  # Only 1 non-zero training weight when validating on 4 rows.
        cvm = CVMatrix(ddof=1, backend="jax")
        cvm.fit(X, Y, weights)
        with pytest.raises(ValueError, match="greater than `ddof`"):
            cvm.training_XTX_XTY(np.arange(4))
        (XTWX, _), _ = jax.jit(cvm.training_XTX_XTY)(jnp.arange(4))  # Deferred.
        assert XTWX.shape == (4, 4)

    def test_jax_traced_weights(self):
        """
        Tests sample weights traced by jax.vmap/jax.jit, e.g., per-dataset weights when
        vmapping over datasets: CVMatrix is fitted inside the traced function. All
        training matrices and statistics must match eager per-dataset computation,
        with centering and scaling and ddof 0 and 1, for: per-dataset validation
        indices vmapped over folds under jit(vmap); shared concrete indices closed
        over; per-fold calls with concrete indices inside a vmap over the weights; and
        concrete weights closed over inside jax.jit.
        """
        if getattr(self, "backend", "numpy") != "jax":
            pytest.skip("JAX-specific trace test; runs under the jax parametrization.")
        pytest.importorskip("jax")
        import jax
        import jax.numpy as jnp

        # As for users, JAX's 64-bit mode must be on before tracing starts, because
        # jax.jit fixes the input dtypes then; a CVMatrix constructed inside the traced
        # function enables it too late.
        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        B, N, K, M, n_folds = 4, 30, 6, 2, 5
        X = rng.standard_normal((B, N, K))
        Y = rng.standard_normal((B, N, M))
        W = rng.uniform(0.1, 2.0, (B, N))
        val = np.stack([rng.permutation(N).reshape(n_folds, -1) for _ in range(B)])
        shared_val = val[0]
        methods = (
            "training_XTX",
            "training_XTY",
            "training_XTX_XTY",
            "training_statistics",
        )

        for ddof in (0, 1):

            def fitted(x, y, w, ddof=ddof):
                cvm = CVMatrix(ddof=ddof, backend="jax")
                cvm.fit(x, y, w)
                return cvm

            def all_methods(cvm, val_indices):
                return tuple(getattr(cvm, m)(val_indices) for m in methods)

            def check(result, b, val_indices, index=()):
                expected = self.leaves(
                    all_methods(fitted(X[b], Y[b], W[b]), val_indices)
                )
                actual = self.leaves(result)
                assert len(actual) == len(expected)
                for a, e in zip(actual, expected):
                    if e is None:
                        assert a is None
                    else:
                        assert_allclose(
                            np.asarray(a)[index], e, rtol=1e-10, atol=1e-10
                        )

            # Per-dataset weights and validation indices, folds vmapped, under jit.
            def per_dataset(x, y, w, vi):
                cvm = fitted(x, y, w)
                return jax.vmap(lambda v: all_methods(cvm, v))(vi)

            result = jax.jit(jax.vmap(per_dataset))(X, Y, W, val)
            for b, f in product(range(B), range(n_folds)):
                check(result, b, val[b, f], (b, f))

            # Per-dataset weights; shared concrete indices closed over.
            def shared_indices(x, y, w):
                cvm = fitted(x, y, w)
                return jax.vmap(lambda v: all_methods(cvm, v))(jnp.asarray(shared_val))

            result = jax.jit(jax.vmap(shared_indices))(X, Y, W)
            for b, f in product(range(B), range(n_folds)):
                check(result, b, shared_val[f], (b, f))

            # Per-fold calls with concrete indices inside a vmap over the weights.
            def per_fold(x, y, w):
                cvm = fitted(x, y, w)
                return tuple(all_methods(cvm, v) for v in shared_val)

            result = jax.vmap(per_fold)(X, Y, W)
            for b, f in product(range(B), range(n_folds)):
                check(result[f], b, shared_val[f], (b,))

            # Concrete weights closed over inside jax.jit, with concrete indices.
            def closed_over(x, y):
                cvm = fitted(x, y, W[0])
                return tuple(all_methods(cvm, v) for v in shared_val)

            result = jax.jit(closed_over)(X[0], Y[0])
            for f in range(n_folds):
                check(result[f], 0, shared_val[f])

    def test_jax_traced_weights_validation(self):
        """
        Tests which weight checks run when CVMatrix is traced by jax.jit/jax.vmap.
        Concrete weights and validation indices are checked as in eager execution, also
        inside jax.jit: negative weights closed over, and degenerate folds both with
        weights closed over and with a model fitted eagerly. Only traced values are not
        checked (the caller validates them before tracing): negative or degenerate
        traced weights, also when given as a list of traced scalars.
        """
        if getattr(self, "backend", "numpy") != "jax":
            pytest.skip("JAX-specific trace test; runs under the jax parametrization.")
        pytest.importorskip("jax")
        import jax

        # As for users, JAX's 64-bit mode must be on before tracing starts, because
        # jax.jit fixes the input dtypes then; a CVMatrix constructed inside the traced
        # function enables it too late.
        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        B, N, K, M = 3, 12, 4, 2
        X = rng.standard_normal((B, N, K))
        Y = rng.standard_normal((B, N, M))
        W = rng.uniform(0.1, 2.0, (B, N))
        val_indices = np.arange(4)

        def fit_and_train(x, y, w):
            cvm = CVMatrix(ddof=1, backend="jax")
            cvm.fit(x, y, w)
            return cvm.training_XTX_XTY(val_indices)

        negative = W[0].copy()
        negative[5] = -1.0
        with pytest.raises(ValueError, match="Weights must be non-negative."):
            jax.jit(lambda x, y: fit_and_train(x, y, negative))(X[0], Y[0])
        (XTWX, _), _ = jax.vmap(fit_and_train)(X, Y, -W)  # Traced: not checked.
        assert XTWX.shape == (B, K, K)
        (XTWX, _), _ = jax.jit(
            lambda x, s: fit_and_train(x, Y[0], [s] * N)  # A list of traced scalars.
        )(X[0], 1.0)
        assert XTWX.shape == (K, K)

        # No non-zero training weights, and fewer than ddof + 1 of them.
        no_train = np.zeros(N)
        no_train[val_indices] = 1.0
        one_train = no_train.copy()
        one_train[N - 1] = 1.0
        for weights, match in (
            (no_train, "greater than zero"),
            (one_train, "greater than `ddof`"),
        ):
            with pytest.raises(ValueError, match=match):
                fit_and_train(X[0], Y[0], weights)
            with pytest.raises(ValueError, match=match):
                jax.jit(lambda x, y, weights=weights: fit_and_train(x, y, weights))(
                    X[0], Y[0]
                )
            fitted = CVMatrix(ddof=1, backend="jax")
            fitted.fit(X[0], Y[0], weights)
            with pytest.raises(ValueError, match=match):
                jax.jit(lambda fitted=fitted: fitted.training_XTX_XTY(val_indices))()
            batch = np.broadcast_to(weights, (B, N))
            (XTWX, _), _ = jax.vmap(fit_and_train)(X, Y, batch)  # Traced: deferred.
            assert XTWX.shape == (B, K, K)

    def test_empty_training_set(self):
        """
        Tests that a fold whose training set is empty raises a ValueError when the
        training statistics are needed, both unweighted and weighted, instead of
        silently giving NaN means in the unweighted case.
        """
        X = self.load_X()[:20, :5]
        Y = self.load_Y(["Protein", "Moisture"])[:20]
        all_indices = np.arange(X.shape[0])
        for weights in (None, self.load_weights(random=True)[:20]):
            fast = CVMatrix(
                center_X=True, center_Y=True, scale_X=False, scale_Y=False,
                ddof=0, backend=self.backend,
            )
            fast.fit(X, Y, weights)
            for method in ("training_statistics", "training_XTX", "training_XTX_XTY"):
                with pytest.raises(ValueError, match="greater than zero"):
                    getattr(fast, method)(all_indices)

    def test_failed_fit_leaves_instance_unchanged(self):
        """
        Tests that a fit that raises on negative weights leaves a previously fitted
        instance unchanged, instead of mixing the new X, Y, and weights with the
        previous matrix products and statistics.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        val_indices = np.arange(0, X.shape[0], 7)
        fast = CVMatrix(backend=self.backend)
        fast.fit(X, Y, weights)
        state = dict(vars(fast))
        (XTWX, XTWY), stats = fast.training_XTX_XTY(val_indices)
        negative = weights.copy()
        negative[0] = -1.0
        with pytest.raises(ValueError, match="Weights must be non-negative."):
            fast.fit(2 * X, 2 * Y, negative)
        assert vars(fast).keys() == state.keys()
        assert all(vars(fast)[k] is v for k, v in state.items())
        (XTWX_after, XTWY_after), stats_after = fast.training_XTX_XTY(val_indices)
        assert_allclose(XTWX_after, XTWX)
        assert_allclose(XTWY_after, XTWY)
        self.check_equivalent_stats(stats, stats_after)

    def test_validate_folds(self):
        """
        Tests `validate_folds`, the host-side pre-flight check for folds and weights
        that will be traced. It needs no fit, accepts valid folds in every supported
        form, agrees fold by fold with the eager checks of the training methods, and
        reports every failing fold, including out-of-range validation indices, in one
        ValueError.
        """
        rng = np.random.default_rng(0)
        N = 12
        X, Y = rng.random((N, 4)), rng.random((N, 2))
        weights = rng.uniform(0.1, 2.0, N)
        weights[:5] = 0.0  # Only rows 5 to 11 have non-zero weights.
        unfitted = CVMatrix(ddof=1, backend=self.backend)
        p = Partitioner(np.arange(N) % 3)
        for folds in (
            p,
            p.folds_dict,
            list(p.folds_dict.values()),
            np.stack(list(p.folds_dict.values())),
        ):
            unfitted.validate_folds(folds, N, weights)
            unfitted.validate_folds(folds, N)

        # Fold by fold, validate_folds raises exactly when eager execution does.
        fitted = CVMatrix(ddof=1, backend=self.backend)
        fitted.fit(X, Y, weights)
        for val_indices, match in (
            (np.arange(5, N), "greater than zero"),  # No non-zero training weights.
            (np.arange(6, N), "greater than `ddof`"),  # Only 1 (row 5).
            (np.arange(7, N), None),  # 2 (rows 5 and 6).
            (np.array([0, 1]), None),
        ):
            if match is None:
                fitted.training_statistics(val_indices)
                unfitted.validate_folds([val_indices], N, weights)
                continue
            with pytest.raises(ValueError, match=match):
                fitted.training_statistics(val_indices)
            with pytest.raises(ValueError, match=match):
                unfitted.validate_folds([val_indices], N, weights)

        # Every failing fold is reported, with weights and without.
        folds = {
            "zero": np.arange(5, N),
            "ddof": np.arange(6, N),
            "range": np.array([3, N]),
            "ok": np.array([0, 1]),
        }
        with pytest.raises(ValueError) as excinfo:
            unfitted.validate_folds(folds, N, weights)
        message = str(excinfo.value)
        assert "Fold 'zero'" in message and "Fold 'ddof'" in message
        assert "Fold 'range'" in message and "Fold 'ok'" not in message
        with pytest.raises(ValueError) as excinfo:
            unfitted.validate_folds([np.arange(N), np.arange(N - 1), np.arange(2)], N)
        message = str(excinfo.value)
        assert "Fold 0:" in message and "greater than zero" in message
        assert "Fold 1:" in message and "greater than `ddof`" in message
        assert "Fold 2:" not in message

        # Without centering and scaling, only the index range is checked.
        plain = CVMatrix(
            center_X=False, center_Y=False, scale_X=False, scale_Y=False,
            backend=self.backend,
        )
        plain.validate_folds([np.arange(N)], N, weights)
        with pytest.raises(ValueError, match="Fold 0: the validation indices"):
            plain.validate_folds([np.array([-1])], N)

        # Weights are checked after the cast to `dtype`, like in `fit`.
        with pytest.raises(ValueError, match="has 11 elements"):
            unfitted.validate_folds(p, N, weights[:-1])
        negative = weights.copy()
        negative[6] = -1.0
        with pytest.raises(ValueError, match="Weights must be non-negative."):
            unfitted.validate_folds(p, N, negative)
        tiny = weights.copy()
        tiny[6] = -1e-60  # -0.0 as float32.
        CVMatrix(ddof=1, dtype=np.float32, backend=self.backend).validate_folds(
            p, N, tiny
        )

    def test_weights_validated_after_cast(self):
        """
        Tests that the weights are checked for negative values after they are cast to
        `dtype`, so that both backends, and jax.jit with concrete weights closed over,
        agree: -1e-60 becomes -0.0 as float32, which is accepted, but stays negative
        as float64, which is rejected.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        weights[0] = -1e-60

        def fit(x, y, dtype):
            cvm = CVMatrix(dtype=dtype, backend=self.backend)
            cvm.fit(x, y, weights)
            return cvm.weights

        assert float(fit(X, Y, np.float32)[0, 0]) == 0.0
        with pytest.raises(ValueError, match="Weights must be non-negative."):
            fit(X, Y, np.float64)
        if self.backend == "jax":
            import jax

            jax.jit(lambda x, y: fit(x, y, np.float32))(X, Y)
            with pytest.raises(ValueError, match="Weights must be non-negative."):
                jax.jit(lambda x, y: fit(x, y, np.float64))(X, Y)

    def test_compute_matrix_products(self):
        """
        Tests that `compute_XTX` and `compute_XTY` control which dataset-wide matrix
        products `fit` computes, that the training methods needing a product that is
        not computed raise a ValueError, and that the remaining methods give the same
        results as when both products are computed.
        """

        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        folds = self.load_Y(["split"]).squeeze()
        p = Partitioner(folds)
        val_indices = p.get_validation_indices(next(iter(p.folds_dict)))
        reference = CVMatrix(backend=self.backend)
        reference.fit(X, Y, weights)
        for compute_XTX, compute_XTY in product([True, False], repeat=2):
            fast = CVMatrix(
                backend=self.backend, compute_XTX=compute_XTX, compute_XTY=compute_XTY
            )
            fast.fit(X, Y, weights)
            assert (fast.XTX is not None) == compute_XTX
            assert (fast.XTY is not None) == compute_XTY
            allowed = {
                "training_XTX": compute_XTX,
                "training_XTY": compute_XTY,
                "training_XTX_XTY": compute_XTX and compute_XTY,
                "training_statistics": True,
            }
            for method, is_allowed in allowed.items():
                if not is_allowed:
                    with pytest.raises(ValueError, match="was not computed by `fit`"):
                        getattr(fast, method)(val_indices)
                    continue
                actual = self.leaves(getattr(fast, method)(val_indices))
                desired = self.leaves(getattr(reference, method)(val_indices))
                assert len(actual) == len(desired)
                for a, d in zip(actual, desired):
                    if d is None:
                        assert a is None
                    else:
                        assert_allclose(a, d)

        # The check is on the fitted products, not the flags, so changing a flag after
        # `fit` still raises the documented ValueError.
        fast = CVMatrix(backend=self.backend, compute_XTX=False, compute_XTY=False)
        fast.fit(X, Y, weights)
        fast.compute_XTX = fast.compute_XTY = True
        for method in ("training_XTX", "training_XTY", "training_XTX_XTY"):
            with pytest.raises(ValueError, match="was not computed by `fit`"):
                getattr(fast, method)(val_indices)

    def test_refit_resets_Y_derived_state(self):
        """
        Tests that re-fitting replaces all state derived from `Y` instead of keeping
        it from the previous fit: re-fitting without `Y` resets `XTY`, `WY`, `sum_Y`,
        `sq_Y`, and `sum_sq_Y` to `None`, and re-fitting with weights that do not need
        `WY` resets `WY` to `None` instead of keeping the previous `Y`.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        Y_derived = ("XTY", "WY", "sum_Y", "sq_Y", "sum_sq_Y")
        fast = CVMatrix(backend=self.backend)
        fast.fit(X, Y, weights)
        assert all(getattr(fast, attr) is not None for attr in Y_derived)
        fast.fit(X, None, weights)
        assert all(getattr(fast, attr) is None for attr in Y_derived)

        fast = CVMatrix(
            center_X=False, center_Y=False, scale_X=False, scale_Y=False,
            backend=self.backend,
        )
        fast.fit(X, Y)  # Unweighted: `WY` is `Y` itself.
        assert fast.WY is fast.Y
        fast.fit(X, Y, weights)  # Weighted, but no statistics of `Y` are needed.
        assert fast.WY is None

    def test_get_val_matrices_returns_only_requested(self):
        """
        Tests that `_get_val_matrices` gathers only the requested validation set
        matrices, returning `None` for the others, that the requested ones equal the
        validation rows of `WX`, `X`, `WY`, and `Y`, and that without weights a
        weighted and unweighted matrix requested together share a single gather.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        val_indices = np.arange(0, X.shape[0], 7)
        for weights in (None, self.load_weights(random=True)):
            fast = CVMatrix(backend=self.backend)
            fast.fit(X, Y, weights)
            w = np.ones((X.shape[0], 1)) if weights is None else weights[:, None]
            expected = (X * w, X, Y * w, Y)
            for flags in product([True, False], repeat=4):
                result = fast._get_val_matrices(
                    val_indices,
                    return_X_val=flags[0],
                    return_X_val_unweighted=flags[1],
                    return_Y_val=flags[2],
                    return_Y_val_unweighted=flags[3],
                )
                for requested, mat_val, mat in zip(flags, result, expected):
                    if requested:
                        assert_allclose(mat_val, mat[val_indices])
                    else:
                        assert mat_val is None
                if weights is None:
                    if flags[0] and flags[1]:
                        assert result[0] is result[1]
                    if flags[2] and flags[3]:
                        assert result[2] is result[3]

    def test_pickle(self):
        """
        Tests that a fitted CVMatrix survives a round trip through the standard
        library's pickle and through copy.deepcopy, which uses the same protocol, with
        the same settings and results, although it holds the array namespace module
        `xp`, which cannot be pickled.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        val_indices = np.arange(0, X.shape[0], 7)
        fast = CVMatrix(backend=self.backend, compute_XTX=False)
        fast.fit(X, Y, weights)
        XTY, stats = fast.training_XTY(val_indices)
        # Only bytes pickled by this test itself are unpickled, so Bandit's B301
        # concern about deserializing untrusted data does not apply.
        unpickled = pickle.loads(pickle.dumps(fast))  # nosec B301
        for clone in (unpickled, deepcopy(fast)):
            assert type(clone) is CVMatrix
            assert clone.xp is fast.xp
            assert clone.XTX is None and not clone.compute_XTX
            clone_XTY, clone_stats = clone.training_XTY(val_indices)
            assert_allclose(clone_XTY, XTY)
            self.check_equivalent_stats(stats, clone_stats)

    def test_process_pool(self):
        """
        Tests that a fitted CVMatrix can be used from worker processes of
        concurrent.futures.ProcessPoolExecutor, which pickles with the standard
        library. The workers are started with `spawn`, i.e., as fresh interpreters (the
        default on Windows and macOS). With the jax backend, this also checks that
        float64 data stays float64 in a fresh process, where JAX's 64-bit mode is
        initially disabled.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        p = Partitioner(self.load_Y(["split"]).squeeze())
        val_indices = [p.get_validation_indices(fold) for fold in p.folds_dict]
        fast = CVMatrix(backend=self.backend)
        fast.fit(X, Y, weights)
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            results = list(executor.map(fast.training_XTX_XTY, val_indices))
        for indices, ((XTX, XTY), stats) in zip(val_indices, results):
            (expected_XTX, expected_XTY), expected_stats = fast.training_XTX_XTY(
                indices
            )
            assert np.asarray(XTX).dtype == np.asarray(XTY).dtype == np.float64
            assert_allclose(XTX, expected_XTX, rtol=1e-12)
            assert_allclose(XTY, expected_XTY, rtol=1e-12)
            self.check_equivalent_stats(expected_stats, stats, atol=1e-12)

    def test_naive_hadamard_vs_matmul(self):
        """
        Tests if the NaiveCVMatrix model gives the same result when using Hadamard
        multiplication, respectively, matrix multiplication.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        folds = self.load_Y(["split"]).squeeze()

        # Subset the data to speed up testing as storing the diagonal matrix W is
        # both memory and time consuming. Computing the weighted matrix product with
        # matrix multiplication is also extremely slow.
        subset_indices = self.subset_indices(folds)
        X = X[subset_indices, :]
        Y = Y[subset_indices, :]
        weights = weights[subset_indices]
        folds = folds[subset_indices]

        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        use_weights = [False, True]
        ddofs = [0, 1]
        for center_X, center_Y, scale_X, scale_Y, use_w, ddof in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys, use_weights, ddofs
        ):
            if use_w:
                cur_weights = self.randomly_zero_weights(weights)
            else:
                continue
            naive_hadamard = self.fit_naive(
                X,
                Y,
                cur_weights,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
                copy=True,
                fast_weight_computation=True,  # Use Hadamard multiplication
            )
            naive_matmul = self.fit_naive(
                X,
                Y,
                cur_weights,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
                copy=True,
                fast_weight_computation=False,  # Use matrix multiplication
            )
            p = Partitioner(folds)
            self.check_equivalent_matrices(naive_hadamard, naive_matmul, p)

    def test_weights_less_than_zero(self):
        """
        Tests that a ValueError is raised when a weight is less than zero.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        weights = self.load_weights(random=True)
        weights[0] = -1.0
        match_str = "Weights must be non-negative."
        with pytest.raises(ValueError, match=match_str):
            self.fit_naive(
                X,
                Y,
                weights,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
        with pytest.raises(ValueError, match=match_str):
            self.fit_fast(
                X,
                Y,
                weights,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )

    def test_ddof_greater_than_nonzero_weights(self):
        """
        Tests that the NaiveCVMatrix and CVMatrix models raise an error when the
        delta degrees of freedom (ddof) is greater than the number of non-zero weights
        in the training set.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 2
        weights = self.load_weights(random=True)
        unique_folds = np.unique(folds)
        weights[2:] = 0
        naive, fast, p = self.fit_models(
            X,
            Y,
            weights,
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            np.float64,
        )
        # Find a training partition with exactly two non-zero weights.
        double_nonzero_training_fold = np.setdiff1d(unique_folds, folds[:2])[0]
        double_nonzero_validation_indices = p.get_validation_indices(
            double_nonzero_training_fold
        )
        double_nonzero_training_indices = np.concatenate(
            [
                p.get_validation_indices(k)
                for k in p.folds_dict
                if k != double_nonzero_training_fold
            ]
        )
        match_str = (
            "The number of non-zero weights in the training set must be greater than "
            "`ddof`."
        )
        with pytest.raises(ValueError, match=match_str):
            fast.training_XTX_XTY(double_nonzero_validation_indices)
        if center_X or scale_X:
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTX(double_nonzero_validation_indices)
        else:
            fast.training_XTX(double_nonzero_validation_indices)
        with pytest.raises(ValueError, match=match_str):
            fast.training_XTY(double_nonzero_validation_indices)
        with pytest.raises(ValueError, match=match_str):
            naive.training_XTX_XTY(double_nonzero_training_indices)
        if center_X or scale_X:
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTX(double_nonzero_training_indices)
        else:
            naive.training_XTX(double_nonzero_training_indices)
        with pytest.raises(ValueError, match=match_str):
            naive.training_XTY(double_nonzero_training_indices)

    def test_train_zeros_weights_preprocessing(self):
        """
        Tests if NaiveCVMatrix and CVMatrix models raise errors when all folds except
        one contain all weights equal to zero, and preprocessing is applied.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        ddof = 0
        weights = self.load_weights(random=True)
        unique_folds = np.unique(folds)
        # Set all weights to zero except for the first fold.
        for fold in unique_folds[1:]:
            weights = self.zero_weights_in_fold(weights, fold)
        for center_X, center_Y, scale_X, scale_Y in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys
        ):
            if not (center_X or center_Y or scale_X or scale_Y):
                # If no preprocessing is applied, the models should not raise an error.
                continue
            naive, fast, p = self.fit_models(
                X,
                Y,
                weights,
                folds,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
            match_str = (
                "The number of non-zero weights in the training set must be greater "
                "than zero."
            )
            fold_with_zero_train_w = unique_folds[0]
            validation_indices = p.get_validation_indices(fold_with_zero_train_w)
            training_indices = np.concatenate(
                [
                    p.get_validation_indices(k)
                    for k in p.folds_dict
                    if k != fold_with_zero_train_w
                ]
            )
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTX_XTY(validation_indices)
            if center_X or scale_X:
                with pytest.raises(ValueError, match=match_str):
                    fast.training_XTX(validation_indices)
            else:
                fast.training_XTX(validation_indices)
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTY(validation_indices)
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTX_XTY(training_indices)
            if center_X or scale_X:
                with pytest.raises(ValueError, match=match_str):
                    naive.training_XTX(training_indices)
            else:
                naive.training_XTX(training_indices)
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTY(training_indices)

    def test_train_zeros_weights(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when all folds except one have all weights equal to zero.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_X = False
        center_Y = False
        scale_X = False
        scale_Y = False
        ddof = 0
        weights = self.load_weights(random=True)
        unique_folds = np.unique(folds)
        # Set all weights to zero except for the first fold.
        for fold in unique_folds[1:]:
            weights = self.zero_weights_in_fold(weights, fold)
        naive, fast, p = self.fit_models(
            X,
            Y,
            weights,
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            np.float64,
        )
        self.check_equivalent_matrices(naive, fast, p)

    def test_val_zeros_weights(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when one fold has all weights equal to zero.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        weights = self.load_weights(random=True)
        weights = self.zero_weights_in_fold(weights, folds[0])
        naive, fast, p = self.fit_models(
            X,
            Y,
            weights,
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            np.float64,
        )
        self.check_equivalent_matrices(naive, fast, p)

    def test_ones_weights(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when all weights are equal to one.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        weights = self.load_weights(random=False)
        naive, fast, p = self.fit_models(
            X,
            Y,
            weights,
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            np.float64,
        )
        naive_unweighted, fast_unweighted, _ = self.fit_models(
            X,
            Y,
            None,
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            np.float64,
        )
        self.check_equivalent_matrices(naive_unweighted, fast, p)
        self.check_equivalent_matrices(naive, fast_unweighted, p)
        self.check_equivalent_matrices(naive, fast, p)

    def test_switch_matrices(self):
        """
        Tests that the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when switching between different matrices.
        """
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([5, 4, 3, 2, 1])
        folds = np.array([0, 0, 1, 1, 2])
        weights = np.array([17, 19, 23, 29, 31])
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        naive, fast, p = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        self.check_equivalent_matrices(naive, fast, p)
        weights = None
        new_naive = self.fit_naive(
            Y, X, weights, center_X, center_Y, scale_X, scale_Y, ddof
        )
        fast.fit(Y, X, weights)
        self.check_equivalent_matrices(new_naive, fast, p)

    def test_constant_columns(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent `X` or `Y` or both contain constant columns.
        """
        X = self.load_X()[:, :5]
        Y = self.load_Y(["Protein", "Moisture"])
        folds = self.load_Y(["split"]).squeeze()
        center_X = False
        center_Y = False
        scale_X = True
        scale_Y = True
        weights = self.random_weights
        ddof = 1
        for i in range(3):
            X = X.copy()
            Y = Y.copy()
            if i == 0:
                X[:, 0] = 1.0
            elif i == 1:
                Y[:, 0] = 1.0
            else:
                X[:, 0] = 1.0
                Y[:, 0] = 1.0
            naive, fast, p = self.fit_models(
                X,
                Y,
                weights,
                folds,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
            self.check_equivalent_matrices(naive, fast, p)

    def test_no_second_dimension_provided(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when `X` or `Y` do not have a second dimension. This tests the
        functionality of CVMatrix._init_mat.
        """
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([5, 4, 3, 2, 1])
        weights = np.array([2, 4, 6, 8, 10])
        folds = np.array([0, 0, 1, 1, 2])
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        naive, fast, p = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof, np.float64
        )
        self.check_equivalent_matrices(naive, fast, p)
        XTXs_XTYs, stats = zip(
            *[
                naive.training_XTX_XTY(
                    np.concatenate(
                        [p.get_validation_indices(k) for k in p.folds_dict if k != fold]
                    )
                )
                for fold in p.folds_dict
            ]
        )
        XTXs, XTYs = zip(*XTXs_XTYs)
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
        fast.fit(X, Y, weights)
        expanded_XTXs_expanded_XTYs, expanded_stats = zip(
            *[
                fast.training_XTX_XTY(p.get_validation_indices(fold))
                for fold in p.folds_dict
            ]
        )
        expanded_XTXs, expanded_XTYs = zip(*expanded_XTXs_expanded_XTYs)
        assert_allclose(XTXs, expanded_XTXs)
        assert_allclose(XTYs, expanded_XTYs)
        for naive_stats, fast_stats in zip(stats, expanded_stats):
            self.check_equivalent_stats(naive_stats, fast_stats)

    def test_no_response_variables(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when no response variables are provided.
        """
        X = self.load_X()[:, :5]
        Y = None
        folds = self.load_Y(["split"]).squeeze()
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        weights = self.random_weights
        naive, fast, p = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof, np.float64
        )
        self.check_equivalent_matrices(naive, fast, p)

    def test_dtype(self):
        """
        Tests that different dtypes can be used and that the output preserves the
        dtype.
        """
        if getattr(self, "backend", "numpy") == "jax":
            pytest.skip(
                "NumPy-specific dtype coverage (incl. float16/float128); the jax backend "
                "supports only float32/float64."
            )
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([5, 4, 3, 2, 1])
        folds = np.array([0, 0, 1, 1, 2])
        weights = np.array([3, 6, 9, 12, 15])
        center_Xs = [True, False]
        center_Ys = [True, False]
        scale_Xs = [True, False]
        scale_Ys = [True, False]
        ddofs = [0, 1]
        use_weights = [False, True]
        dtypes = [np.float16, np.float32, np.float64]
        if not sys.platform.startswith("win") and not sys.platform.startswith("darwin"):
            # Windows and MacOS do not support float128
            dtypes.append(np.float128)
        for center_X, center_Y, scale_X, scale_Y, ddof, dtype, use_w in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys, ddofs, dtypes, use_weights
        ):
            w = weights if use_w else None
            naive, fast, p = self.fit_models(
                X, Y, w, folds, center_X, center_Y, scale_X, scale_Y, ddof, dtype
            )
            naive_XTXs_naive_XTYs, _naive_stats_all_folds = zip(
                *[
                    naive.training_XTX_XTY(
                        np.concatenate(
                            [
                                p.get_validation_indices(k)
                                for k in p.folds_dict
                                if k != fold
                            ]
                        )
                    )
                    for fold in p.folds_dict
                ]
            )
            naive_XTXs, naive_XTYs = zip(*naive_XTXs_naive_XTYs)
            fast_XTXs_fast_XTYs, _fast_stats_all_folds = zip(
                *[
                    fast.training_XTX_XTY(p.get_validation_indices(fold))
                    for fold in p.folds_dict
                ]
            )
            fast_XTXs, fast_XTYs = zip(*fast_XTXs_fast_XTYs)
            for naive_XTX, fast_XTX in zip(naive_XTXs, fast_XTXs):
                assert naive_XTX.dtype == dtype
                assert fast_XTX.dtype == dtype
            for naive_XTY, fast_XTY in zip(naive_XTYs, fast_XTYs):
                assert naive_XTY.dtype == dtype
                assert fast_XTY.dtype == dtype

    def test_copy(self):
        """
        Tests that the copy parameter works as expected.
        """
        if getattr(self, "backend", "numpy") == "jax":
            pytest.skip(
                "`copy`/`np.shares_memory` are NumPy-specific; the jax backend always "
                "transfers inputs to device arrays."
            )
        dtype = np.float64
        X = np.array([1, 2, 3, 4, 5]).astype(dtype)
        Y = np.array([5, 4, 3, 2, 1]).astype(dtype)
        folds = np.array([0, 0, 1, 1, 2])
        weights = np.array([2, 5, 7, 11, 13])
        center_X = False
        center_Y = False
        scale_X = False
        scale_Y = False
        ddof = 1
        for copy in [True, False]:
            naive, fast, p = self.fit_models(
                X,
                Y,
                weights,
                folds,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                dtype,
                copy,
            )
            self.check_equivalent_matrices(naive, fast, p)
            if copy:
                assert not np.shares_memory(naive.X, X)
                assert not np.shares_memory(naive.Y, Y)
                assert not np.shares_memory(fast.X, X)
                assert not np.shares_memory(fast.Y, Y)
            else:
                assert np.shares_memory(naive.X, X)
                assert np.shares_memory(naive.Y, Y)
                assert np.shares_memory(fast.X, X)
                assert np.shares_memory(fast.Y, Y)

    def test_errors(self):
        """
        Tests that errors are raised when expected.
        """
        X = np.array([1, 2, 3, 4, 5])
        Y = None
        folds = np.array([0, 0, 1, 1, 2])
        weights = np.array([37, 41, 43, 47, 53])
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        ddof = 1
        naive, fast, p = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        split_0_val_indices = p.get_validation_indices(0)
        split_0_train_indices = np.concatenate(
            [p.get_validation_indices(k) for k in p.folds_dict if k != 0]
        )
        error_msg = "Response variables `Y` are not provided."
        with pytest.raises(ValueError, match=error_msg):
            naive.training_XTX_XTY(split_0_train_indices)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTX_XTY(split_0_val_indices)
        with pytest.raises(ValueError, match=error_msg):
            naive.training_XTY(split_0_train_indices)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTY(split_0_val_indices)
        error_msg = "At least one of `return_XTX` and `return_XTY` must be True."
        with pytest.raises(ValueError, match=error_msg):
            naive._training_matrices(False, False, split_0_train_indices)
        with pytest.raises(ValueError, match=error_msg):
            fast._training_matrices(False, False, split_0_val_indices)
        invalid_split = 3
        error_msg = f"Fold {invalid_split} not found."
        naive, fast, p = self.fit_models(
            X, X, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        with pytest.raises(ValueError, match=error_msg):
            p.get_validation_indices(invalid_split)

    def test_statistics_cvmatrix_methods(self):
        """
        Tests if the statistics computed by different CVMatrix methods are equivalent.
        """
        X = self.load_X()[:, :5]
        Ys = [None, self.load_Y(["Protein", "Moisture"])]
        folds = self.load_Y(["split"]).squeeze()
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        ddofs = [0, 1]
        use_weights = [False, True]
        for center_X, center_Y, scale_X, scale_Y, use_w, ddof, Y in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys, use_weights, ddofs, Ys
        ):
            diagnostic_msg = (
                f"center_X: {center_X}, center_Y: {center_Y}, "
                f"scale_X: {scale_X}, scale_Y: {scale_Y}, "
                f"ddof: {ddof}, use_weights: {use_w}, use_Y: {Y is not None}"
            )
            if use_w:
                weights = self.randomly_zero_weights(self.load_weights(random=True))
            else:
                weights = None
            fast = self.fit_fast(
                X,
                Y,
                weights,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
            p = self.instantiate_p(folds)
            print(diagnostic_msg)
            for val_indices in p.folds_dict.values():
                stats1 = fast.training_statistics(val_indices)
                if Y is not None:
                    _, stats2 = fast.training_XTX_XTY(val_indices)
                    self.check_equivalent_stats(
                        stats1,
                        stats2,
                        err_msg="Statistics from training_statistics and "
                        "training_XTX_XTY methods are not equivalent." + diagnostic_msg,
                    )
                    _, stats3 = fast.training_XTY(val_indices)
                    self.check_equivalent_stats(
                        stats1,
                        stats3,
                        err_msg="Statistics from training_statistics and "
                        "training_XTY methods are not equivalent." + diagnostic_msg,
                    )
                _, stats4 = fast.training_XTX(val_indices)
                self.check_equivalent_stats(
                    stats1,
                    stats4,
                    err_msg="Statistics from training_statistics and "
                    "training_XTX methods are not equivalent." + diagnostic_msg,
                )

    def test_loocv(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent when using Leave-One-Out Cross-Validation (LOOCV).
        """
        X = self.load_X()[:, :5]
        Ys = [None, self.load_Y(["Protein", "Moisture"])]
        folds = np.arange(X.shape[0])
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        ddofs = [0, 1]
        use_weights = [False, True]
        for center_X, center_Y, scale_X, scale_Y, use_w, ddof, Y in product(
            center_Xs, center_Ys, scale_Xs, scale_Ys, use_weights, ddofs, Ys
        ):
            diagnostic_msg = (
                f"center_X: {center_X}, center_Y: {center_Y}, "
                f"scale_X: {scale_X}, scale_Y: {scale_Y}, "
                f"ddof: {ddof}, use_weights: {use_w}, use_Y: {Y is not None}"
            )
            if use_w:
                weights = self.randomly_zero_weights(self.load_weights(random=True))
            else:
                weights = None
            naive, fast, p = self.fit_models(
                X,
                Y,
                weights,
                folds,
                center_X,
                center_Y,
                scale_X,
                scale_Y,
                ddof,
                np.float64,
            )
            print(diagnostic_msg)
            self.check_equivalent_matrices(naive, fast, p, max_folds=20)
