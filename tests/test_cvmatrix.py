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

import sys
from collections.abc import Hashable
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
        fast = CVMatrix(center_X, center_Y, scale_X, scale_Y, ddof, dtype, copy)
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
