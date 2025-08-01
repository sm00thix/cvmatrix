"""
Contains the CVMatrix class which implements methods for naive computation of training
set kernel matrices in cross-validation using the naive algorithms described in the
paper by Engstrøm and Jensen. The implementation is written using NumPy.

O.-C. G. Engstrøm and M. H. Jensen (2025):
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Tuple, Union

import numpy as np
from numpy import typing as npt

from cvmatrix.cvmatrix import CVMatrix


class NaiveCVMatrix(CVMatrix):
    """
    Implements the naive cross-validation algorithms for kernel matrix-based models such
    as PCA, PCR, PLS, and OLS. The algorithms are described in detail in the paper by
    O.-C. G. Engstrøm and M. H. Jensen:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

    Parameters
    ----------
    center_X : bool, optional, default=True
        Whether to center `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` by subtracting its row of
        column-wise weighted means from each row. The row of column-wise weighted
        means is computed on the training set for each fold to avoid data leakage.

    center_Y : bool, optional, default=True
        Whether to center `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` by subtracting its row of
        column-wise weighted means from each row. The row of column-wise weighted means
        is computed on the training set for each fold to avoid data leakage. This
        parameter is ignored if `Y` is `None`.

    scale_X : bool, optional, default=True
        Whether to scale `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` by dividing each row with
        the row of `X`'s column-wise weighted standard deviations. The row of
        column-wise weighted standard deviations is computed on the training set for
        each fold to avoid data leakage.

    scale_Y : bool, optional, default=True
        Whether to scale `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` by dividing each row with
        the row of `Y`'s column-wise weighted standard deviations. The row of
        column-wise weighted standard deviations is computed on the training set for
        each fold to avoid data leakage. This parameter is ignored if `Y` is `None`.

    ddof : int, optional, default=1
        The delta degrees of freedom used in the computation of the sample standard
        deviation. The default is 1, which corresponds to Bessel's correction for the
        unbiased estimate of the sample standard deviation. If `ddof` is set to 0,
        the population standard deviation is computed instead.

    dtype : np.floating, optional, default=np.float64
        The data type used for the computations. The default is `np.float64`.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.

    fast_weight_computation : bool, optional, default=True
        If `True`, the computation of the weighted matrix products are done by
        computing the hadamard product of the weights and either `X` or `Y`. If
        `False`, the weigthed matrix products are computed by constructing a
        diagonal matrix from the weights and then performing regular matrix
        multiplication. The hadamard or matrix product is always computed over the
        smallest of `X` and `Y` if both XTWX and XTWY are requested to speed up the
        computation. This parameter is ignored if `W` is `None`.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        dtype: np.floating = np.float64,
        copy: bool = True,
        fast_weight_computation: bool = True,
    ) -> None:
        super().__init__(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            ddof=ddof,
            dtype=dtype,
            copy=copy,
        )
        self.W = None
        self.fast_weight_computation = fast_weight_computation

    def fit(
        self,
        X: npt.ArrayLike,
        Y: Union[None, npt.ArrayLike] = None,
        weights: Union[None, npt.ArrayLike] = None,
    ) -> None:
        """
        Loads and stores `X`, `Y`, and `weights` for cross-validation.

        Parameters
        ----------
        X : Array-like of shape (N, K) or (N,)
            Predictor variables.

        Y : None or array-like of shape (N, M) or (N,), optional, default=None
            Response variables. If `None`, subsequent calls to training_XTY and
            training_XTX_XTY will raise a `ValueError`.

        weights : None or array-like of shape (N,) or (N, 1), optional, default=None
            Weights for each sample in `X` and `Y`. If `None`, no weights are used in
            the computations. If provided, the weights must be non-negative.

        Raises
        ------
        ValueError
            If `weights` is provided and contains negative values.
        """
        self.X = self._init_mat(X)
        self.N, self.K = self.X.shape
        if Y is not None:
            self.Y = self._init_mat(Y)
            self.M = self.Y.shape[1]
        if weights is not None:
            self.weights = self._init_mat(weights)
            if np.any(self.weights < 0):
                raise ValueError("Weights must be non-negative.")
            if not self.fast_weight_computation:
                self.W = np.diag(self.weights.squeeze())
        else:
            self.weights = None

    def _compute_training_mat_std(
        self,
        mat: np.ndarray,
        mean_row: np.ndarray,
        weights: Union[np.ndarray, None],
        scale_factor: Union[float, None],
    ) -> np.ndarray:
        """
        Computes the standard deviation of each column in `mat`, weighted by `w`.
        Bessel's correction is applied.
        """
        if weights is None:
            std = mat.std(axis=0, ddof=self.ddof, keepdims=True, mean=mean_row)
        else:
            std = np.sqrt(
                np.sum(weights * (mat - mean_row) ** 2, axis=0, keepdims=True)
                / (scale_factor)
            )
        std[np.abs(std) <= self.resolution] = 1
        return std

    def _training_matrices(
        self, return_XTX: bool, return_XTY: bool, training_indices: npt.NDArray[np.int_]
    ) -> Tuple[
        Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        Tuple[
            Union[np.ndarray, None],
            Union[np.ndarray, None],
            Union[np.ndarray, None],
            Union[np.ndarray, None],
        ],
    ]:
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y is None:
            raise ValueError("Response variables `Y` are not provided.")
        X_train = self.X[training_indices]
        if self.weights is not None:
            w_train = self.weights[training_indices]
            squeezed_w_train = w_train.squeeze()
            if (
                self.center_X
                or self.scale_X
                or (return_XTY and (self.center_Y or self.scale_Y))
            ):
                num_nonzero_weights = np.asarray(
                    np.count_nonzero(w_train), dtype=self.dtype
                )
                if num_nonzero_weights == 0:
                    raise ValueError(
                        "The number of non-zero weights in the training set must be "
                        "greater than zero."
                    )
        else:
            w_train = None
            squeezed_w_train = None

        orig_X_train_mean = X_train_std = orig_Y_train_mean = Y_train_std = None

        if self.center_X or self.scale_X:
            orig_X_train_mean = np.average(
                X_train, axis=0, weights=squeezed_w_train, keepdims=True
            )
            if self.center_X:
                X_train = X_train - orig_X_train_mean
                X_train_mean = 0
            else:
                X_train_mean = orig_X_train_mean

        if (w_train is not None) and (self.scale_X or (return_XTY and self.scale_Y)):
            scale_factor = self._compute_std_divisor(
                np.sum(w_train), num_nonzero_weights
            )
        else:
            scale_factor = None

        if self.scale_X:
            X_train_std = self._compute_training_mat_std(
                mat=X_train,
                mean_row=X_train_mean,
                weights=w_train,
                scale_factor=scale_factor,
            )
            X_train = X_train / X_train_std

        if return_XTY:
            Y_train = self.Y[training_indices]
            if self.center_Y or self.scale_Y:
                orig_Y_train_mean = np.average(
                    Y_train, axis=0, weights=squeezed_w_train, keepdims=True
                )
                if self.center_Y:
                    Y_train = Y_train - orig_Y_train_mean
                    Y_train_mean = 0
                else:
                    Y_train_mean = orig_Y_train_mean

            if self.scale_Y:
                Y_train_std = self._compute_training_mat_std(
                    mat=Y_train,
                    mean_row=Y_train_mean,
                    weights=w_train,
                    scale_factor=scale_factor,
                )
                Y_train = Y_train / Y_train_std

        if w_train is not None:
            if self.fast_weight_computation:
                operator = np.multiply
                w_train_to_use = w_train.T
            else:
                operator = np.matmul
                w_train_to_use = self.W[np.ix_(training_indices, training_indices)]
            X_train_T_W = operator(X_train.T, w_train_to_use)
        else:
            X_train_T_W = X_train.T

        stats_tuple = (
            orig_X_train_mean,
            X_train_std if self.scale_X else None,
            orig_Y_train_mean if return_XTY else None,
            Y_train_std if (return_XTY and self.scale_Y) else None,
        )

        if return_XTX and return_XTY:
            return (X_train_T_W @ X_train, X_train_T_W @ Y_train), stats_tuple
        if return_XTX:
            return X_train_T_W @ X_train, stats_tuple
        return X_train_T_W @ Y_train, stats_tuple
