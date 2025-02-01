"""
Contains the CVMatrix class which implements methods for naive computation of training
set kernel matrices in cross-validation using the naive algorithms described in the
paper by Engstrøm. The implementation is written using NumPy.

Engstrøm, O.-C. G. (2024):
https://arxiv.org/abs/2401.13185

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

from collections.abc import Hashable
from typing import Iterable, Union

import numpy as np
from numpy import typing as npt

from cvmatrix.cvmatrix import CVMatrix


class NaiveCVMatrix(CVMatrix):
    """
    Implements the naive cross-validation algorithms for kernel matrix-based models such
    as PCA, PCR, PLS, and OLS. The algorithms are described in detail in the paper by
    O.-C. G. Engstrøm: https://arxiv.org/abs/2401.13185

    Parameters
    ----------
    folds : Iterable of Hashable with N elements
        An iterable defining cross-validation splits. Each unique value in
        `folds` corresponds to a different fold.

    center_X : bool, optional, default=True
        Whether to center `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage.

    center_Y : bool, optional, default=True
        Whether to center `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage. This parameter is ignored if `Y` is
        `None`.

    scale_X : bool, optional, default=True
        Whether to scale `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage.

    scale_Y : bool, optional, default=True
        Whether to scale `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage. This parameter is ignored if `Y` is `None`.

    dtype : np.floating, optional, default=np.float64
        The data type used for the computations. The default is `np.float64`.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.
    """

    def __init__(
            self,
            folds: Iterable[Hashable],
            center_X: bool = True,
            center_Y: bool = True,
            scale_X: bool = True,
            scale_Y: bool = True,
            dtype: np.floating = np.float64,
            copy: bool = True,
        ) -> None:
        super().__init__(
            folds=folds,
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            dtype=dtype,
            copy=copy
        )

    def fit(self, X: npt.ArrayLike, Y: Union[None, npt.ArrayLike] = None) -> None:
        """
        Loads and stores `X` and `Y` for cross-validation. Computes dataset-wide
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and, if `Y` is not `None`,
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`. If `center_X`, `center_Y`,
        `scale_X`, or `scale_Y` is `True`, the corresponding global statistics are also
        computed.

        Parameters
        ----------
        X : Array-like of shape (N, K) or (N,)
            Predictor variables.
        
        Y : None or array-like of shape (N, M) or (N,), optional, default=None
            Response variables. If `None`, subsequent calls to training_XTY and
            training_XTX_XTY will raise a `ValueError`.
        """
        self.X_total = self._init_mat(X)
        self.N, self.K = self.X_total.shape
        if Y is not None:
            self.Y_total = self._init_mat(Y)
            self.M = self.Y_total.shape[1]

    def _training_matrices(
            self,
            return_XTX: bool,
            return_XTY: bool,
            fold: Hashable
        ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y_total is None:
            raise ValueError("Response variables `Y` are not provided.")
        training_indices = np.concatenate(
            [
                self.folds_dict.get(i)
                for i in self.folds_dict if i != fold
            ]
        )
        X_train = self.X_total[training_indices]
        if self.center_X:
            X_train_mean = X_train.mean(axis=0, keepdims=True)
            X_train = X_train - X_train_mean
        if self.scale_X:
            X_train_std = X_train.std(axis=0, ddof=1, keepdims=True)
            X_train_std[X_train_std <= self.eps] = 1
            X_train = X_train / X_train_std
        if return_XTY:
            Y_train = self.Y_total[training_indices]
            if self.center_Y:
                Y_train_mean = Y_train.mean(axis=0, keepdims=True)
                Y_train = Y_train - Y_train_mean
            if self.scale_Y:
                Y_train_std = Y_train.std(axis=0, ddof=1, keepdims=True)
                Y_train_std[Y_train_std <= self.eps] = 1
                Y_train = Y_train / Y_train_std
        if return_XTX and return_XTY:
            return X_train.T @ X_train, X_train.T @ Y_train
        if return_XTX:
            return X_train.T @ X_train
        return X_train.T @ Y_train
