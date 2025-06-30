"""
This file contains tests for the cvmatrix package. In general, the tests compare the
output of the fast algorithms implemented in the CVMatrix class with the output of the
naive algorithms implemented in the NaiveCVMatrix class, both of which are described in
the article by Engstrøm. Some of the tests are performed on a real dataset of NIR
spectra and ground truth values for 8 different grain varieties, protein, and moisture.
This dataset is publicly available on GitHub and originates from the articles by Dreier
et al. and Engstrøm et al. See the load_data module for more information about the
dataset.

Engstrøm, O.-C. G. (2024):
https://arxiv.org/abs/2401.13185

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import sys
from collections.abc import Hashable
from itertools import product
from typing import Iterable, Union

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

from cvmatrix.cvmatrix import CVMatrix

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
    ) -> tuple[NaiveCVMatrix, CVMatrix]:
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
            folds,
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
            folds,
            center_X,
            center_Y,
            scale_X,
            scale_Y,
            ddof,
            dtype,
            copy,
        )
        return naive, fast

    def fit_fast(
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
        """
        fast = CVMatrix(folds, center_X, center_Y, scale_X, scale_Y, ddof, dtype, copy)
        fast.fit(X, Y, weights)
        return fast

    def fit_naive(
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
        """
        naive = NaiveCVMatrix(
            folds,
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

    def check_equivalent_matrices(
        self,
        naive: NaiveCVMatrix,
        fast: CVMatrix,
        folds: Iterable[Hashable],
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

        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.
        """
        error_msg = (
            f"\n"
            f"fast center_X: {fast.center_X}, center_Y: {fast.center_Y}, "
            f"scale_X: {fast.scale_X}, scale_Y: {fast.scale_Y}"
            f"\nddof: {fast.ddof}, dtype: {fast.dtype}, copy: {fast.copy}, "
            f"weights: {fast.w_total is not None}, "
            f"\n"
            f"\nnaive center_X: {naive.center_X}, center_Y: {naive.center_Y}, "
            f"scale_X: {naive.scale_X}, scale_Y: {naive.scale_Y}"
            f"\nddof: {naive.ddof}, dtype: {naive.dtype}, copy: {naive.copy}, "
            f"weights: {naive.w_total is not None}"
            f"\n"
        )
        for fold in np.unique(folds):
            error_msg = f"Fold: {fold}\n{error_msg}"
            if naive.Y_total is not None:
                # Check if the matrices are equivalent for the training_XTX_XTY method
                # between the NaiveCVMatrix and CVMatrix models.
                fast_XTX, fast_XTY = fast.training_XTX_XTY(fold)
                naive_XTX, naive_XTY = naive.training_XTX_XTY(fold)
                assert_allclose(fast_XTX, naive_XTX, err_msg=error_msg, atol=1e-8)
                assert_allclose(fast_XTY, naive_XTY, err_msg=error_msg, atol=1e-8)
                # Check if the matrices are equivalent for the training_XTX and
                # training_XTY methods between the NaiveCVMatrix and CVMatrix models.
                # Also check if the matrices are equivalent for the training_XTX,
                # training_XTY, and training_XTX_XTY methods.
                direct_naive_XTX = naive.training_XTX(fold)
                direct_fast_XTX = fast.training_XTX(fold)
                direct_naive_XTY = naive.training_XTY(fold)
                direct_fast_XTY = fast.training_XTY(fold)
                assert_allclose(
                    direct_fast_XTX, direct_naive_XTX, err_msg=error_msg, atol=1e-8
                )
                assert_allclose(
                    direct_fast_XTY, direct_naive_XTY, err_msg=error_msg, atol=1e-8
                )
                assert_allclose(direct_fast_XTX, fast_XTX, err_msg=error_msg, atol=1e-8)
                assert_allclose(direct_fast_XTY, fast_XTY, err_msg=error_msg, atol=1e-8)
            else:
                # Check if the matrices are equivalent for the training_XTX method
                # between the NaiveCVMatrix and CVMatrix models.
                naive_XTX = naive.training_XTX(fold)
                fast_XTX = fast.training_XTX(fold)
                assert_allclose(fast_XTX, naive_XTX, err_msg=error_msg, atol=1e-8)

    def test_all_preprocessing_combinations(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent for basic settings.
        """
        X = self.load_X()[:, :5]  # Use only the first 5 variables for faster testing.
        Y = self.load_Y(["Protein", "Moisture"])
        weights = self.load_weights(random=True)
        folds = self.load_Y(["split"]).squeeze()
        assert X.shape[0] == Y.shape[0] == folds.shape[0] == weights.shape[0]
        assert len(np.unique(folds)) == 3
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
                cur_weights = None
            naive, fast = self.fit_models(
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
            self.check_equivalent_matrices(naive, fast, folds)

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
                cur_weights = None
            naive_hadamard = self.fit_naive(
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
                copy=True,
                fast_weight_computation=True,  # Use Hadamard multiplication
            )
            naive_matmul = self.fit_naive(
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
                copy=True,
                fast_weight_computation=False,  # Use matrix multiplication
            )
            self.check_equivalent_matrices(naive_hadamard, naive_matmul, folds)

    def test_weights_less_than_zero(self):
        """
        Tests that a ValueError is raised when a weight is less than zero.
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
        weights[0] = -1.0
        match_str = "Weights must be non-negative."
        with pytest.raises(ValueError, match=match_str):
            self.fit_naive(
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
        with pytest.raises(ValueError, match=match_str):
            self.fit_fast(
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
        # Find a training partition with exactly two non-zero weights.
        double_nonzero_training_fold = np.setdiff1d(unique_folds, folds[:2])[0]
        naive, fast = self.fit_models(
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
        match_str = "The number of non-zero weights in the training set must be greater than `ddof`."
        with pytest.raises(ValueError, match=match_str):
            fast.training_XTX_XTY(double_nonzero_training_fold)
        if center_X or scale_X:
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTX(double_nonzero_training_fold)
        else:
            fast.training_XTX(double_nonzero_training_fold)
        with pytest.raises(ValueError, match=match_str):
            fast.training_XTY(double_nonzero_training_fold)
        with pytest.raises(ValueError, match=match_str):
            naive.training_XTX_XTY(double_nonzero_training_fold)
        if center_X or scale_X:
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTX(double_nonzero_training_fold)
        else:
            naive.training_XTX(double_nonzero_training_fold)
        with pytest.raises(ValueError, match=match_str):
            naive.training_XTY(double_nonzero_training_fold)

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
            naive, fast = self.fit_models(
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
            match_str = "The number of non-zero weights in the training set must be greater than zero."
            fold_with_zero_train_w = unique_folds[0]
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTX_XTY(fold_with_zero_train_w)
            if center_X or scale_X:
                with pytest.raises(ValueError, match=match_str):
                    fast.training_XTX(fold_with_zero_train_w)
            else:
                fast.training_XTX(fold_with_zero_train_w)
            with pytest.raises(ValueError, match=match_str):
                fast.training_XTY(fold_with_zero_train_w)
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTX_XTY(fold_with_zero_train_w)
            if center_X or scale_X:
                with pytest.raises(ValueError, match=match_str):
                    naive.training_XTX(fold_with_zero_train_w)
            else:
                naive.training_XTX(fold_with_zero_train_w)
            with pytest.raises(ValueError, match=match_str):
                naive.training_XTY(fold_with_zero_train_w)

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
        naive, fast = self.fit_models(
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
        self.check_equivalent_matrices(naive, fast, folds)

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
        naive, fast = self.fit_models(
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
        self.check_equivalent_matrices(naive, fast, folds)

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
        naive, fast = self.fit_models(
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
        naive_unweighted, fast_unweighted = self.fit_models(
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
        self.check_equivalent_matrices(naive_unweighted, fast, folds)
        self.check_equivalent_matrices(naive, fast_unweighted, folds)
        self.check_equivalent_matrices(naive, fast, folds)

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
        naive, fast = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        self.check_equivalent_matrices(naive, fast, folds)
        weights = None
        new_naive = self.fit_naive(
            Y, X, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        fast.fit(Y, X, weights)
        self.check_equivalent_matrices(new_naive, fast, folds)

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
            naive, fast = self.fit_models(
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
            self.check_equivalent_matrices(naive, fast, folds)

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
        naive, fast = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof, np.float64
        )
        self.check_equivalent_matrices(naive, fast, folds)
        XTXs, XTYs = zip(
            *[naive.training_XTX_XTY(val_split) for val_split in np.unique(folds)]
        )
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
        fast.fit(X, Y, weights)
        expanded_XTXs, expanded_XTYs = zip(
            *[fast.training_XTX_XTY(val_split) for val_split in np.unique(folds)]
        )
        assert_allclose(XTXs, expanded_XTXs)
        assert_allclose(XTYs, expanded_XTYs)

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
        naive, fast = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof, np.float64
        )
        self.check_equivalent_matrices(naive, fast, folds)

    def test_dtype(self):
        """
        Tests that different dtypes can be used and that the output preserves the
        dtype.
        """
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
            naive, fast = self.fit_models(
                X, Y, w, folds, center_X, center_Y, scale_X, scale_Y, ddof, dtype
            )
            naive_XTXs, naive_XTYs = zip(
                *[naive.training_XTX_XTY(val_split) for val_split in np.unique(folds)]
            )
            fast_XTXs, fast_XTYs = zip(
                *[fast.training_XTX_XTY(val_split) for val_split in np.unique(folds)]
            )
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
            naive, fast = self.fit_models(
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
            self.check_equivalent_matrices(naive, fast, folds)
            if copy:
                assert not np.shares_memory(naive.X_total, X)
                assert not np.shares_memory(naive.Y_total, Y)
                assert not np.shares_memory(fast.X_total, X)
                assert not np.shares_memory(fast.Y_total, Y)
            else:
                assert np.shares_memory(naive.X_total, X)
                assert np.shares_memory(naive.Y_total, Y)
                assert np.shares_memory(fast.X_total, X)
                assert np.shares_memory(fast.Y_total, Y)

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
        naive, fast = self.fit_models(
            X, Y, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )

        error_msg = "Response variables `Y` are not provided."
        with pytest.raises(ValueError, match=error_msg):
            naive.training_XTX_XTY(0)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTX_XTY(0)
        with pytest.raises(ValueError, match=error_msg):
            naive.training_XTY(0)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTY(0)
        error_msg = "At least one of `return_XTX` and `return_XTY` must be True."
        with pytest.raises(ValueError, match=error_msg):
            naive._training_matrices(False, False, 0)
        with pytest.raises(ValueError, match=error_msg):
            fast._training_matrices(False, False, 0)
        invalid_split = 3
        error_msg = f"Fold {invalid_split} not found."
        naive, fast = self.fit_models(
            X, X, weights, folds, center_X, center_Y, scale_X, scale_Y, ddof
        )
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTX_XTY(invalid_split)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTX(invalid_split)
        with pytest.raises(ValueError, match=error_msg):
            fast.training_XTY(invalid_split)
