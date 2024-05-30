"""
TODO: Write module docstring and signature
"""

from typing import Hashable, Iterable, Union

import numpy as np
from numpy import typing as npt


class CVMatrix:
    """
    Implements the fast cross-validation algorithms for kernel matrix-based models
    such as PCA, PCR, PLS, and OLS. The algorithms are based on the following paper by
    O.-C. G. Engstrøm: https://arxiv.org/abs/2401.13185

    Parameters
    ----------
    X : Array-like of shape (N, K) or (N,)
        Predictor variables.
    
    Y : None or array-like of shape (N, M) or (N,), optional, default=None
        Response variables. If `None`, only :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
        will be computed and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` will not be
        computed. This is useful for models such as PCA and PCR.

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

    dtype : data-type, optional, default=np.float64
        The data-type of the arrays used in the computation.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        dtype: npt.DTypeLike = np.float64,
        copy: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.dtype = dtype
        self.copy = copy
        self.X_total = None
        self.Y_total = None
        self.N = None
        self.K = None
        self.M = None
        self.val_index_dict = None
        self.X_total_mean = None
        self.Y_total_mean = None
        self.XTX_total = None
        self.XTY_total = None
        self.sum_X_total = None
        self.sum_Y_total = None
        self.sum_sq_X_total = None
        self.sum_sq_Y_total = None

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
        self.XTX_total = self.X_total.T @ self.X_total
        if Y is not None:
            self.Y_total = self._init_mat(Y)
            self.M = self.Y_total.shape[1]
            self.XTY_total = self.X_total.T @ self.Y_total
        self._init_total_stats()

    def load_cv_splits(self, cv_splits: Iterable[Hashable]) -> None:
        """
        Loads new cross-validation splits.

        Parameters
        ----------
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.
        """
        self._init_val_indices_dict(cv_splits)

    def training_XTX(self, val_idx: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` for a given
        fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        Returns
        -------
        Array of shape (K, K)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        Raises
        ------

        See Also
        --------
        training_XTX_XTY : Returns the training set
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold. This method is
        faster than calling `training_XTX` and `training_XTY` separately.
        """
        return self._training_matrices(True, False, val_idx)

    def training_XTY(self, val_idx: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given
        fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        Array of shape (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Raises
        ------
        ValueError
            If `Y` is `None`.

        See Also
        --------
        training_XTX_XTY : Returns the training set
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold. This method is
        faster than calling `training_XTX` and `training_XTY` separately.
        """
        return self._training_matrices(False, True, val_idx)

    def training_XTX_XTY(self, val_idx: Hashable) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        tuple of arrays of shapes (K, K) and (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.
        """
        return self._training_matrices(True, True, val_idx)

    def _training_matrices(
            self,
            return_XTX: bool,
            return_XTY: bool,
            val_idx: Hashable
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and/or
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold.

        Parameters
        ----------
        return_XTX : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`

        return_XTY : bool, optional, default=False
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        Array of shape (K, K) or (K, M) or tuple of arrays of shapes (K, K) and (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and/or
            training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.
        
        Raises
        ------
        ValueError
            If both `return_XTX` and `return_XTY` are `False` or if `return_XTY` is
            `True` and `Y` is `None`.
        """
        X_train_mean = None
        Y_train_mean = None
        X_train_std = None
        Y_train_std = None
        N_train = None
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y_total is None:
            raise ValueError("Response variables `Y` are not provided.")
        val_indices = self.val_index_dict[val_idx]
        X_val = self.X_total[val_indices]
        if return_XTY:
            Y_val = self.Y_total[val_indices]
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            N_val = val_indices.size
            N_train = self.N - N_val
            N_total_over_N_train = self.N / N_train
            N_val_over_N_train = N_val / N_train
        if self.center_X or self.center_Y or self.scale_X:
            X_train_mean = self._compute_training_mat_mean(
                X_val,
                self.X_total_mean,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if return_XTY and (self.center_X or self.center_Y or self.scale_Y):
            Y_train_mean = self._compute_training_mat_mean(
                Y_val,
                self.Y_total_mean,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if self.scale_X:
            X_train_std = self._compute_training_mat_std(
                X_val,
                X_train_mean,
                self.sum_X_total,
                self.sum_sq_X_total,
                N_train
            )
        if self.scale_Y and return_XTY:
            Y_train_std = self._compute_training_mat_std(
                Y_val,
                Y_train_mean,
                self.sum_Y_total,
                self.sum_sq_Y_total,
                N_train
            )
        if return_XTX and return_XTY:
            return (
                self._training_kernel_matrix(
                    X_val,
                    X_val,
                    X_train_mean,
                    X_train_mean,
                    X_train_std,
                    X_train_std,
                    N_train
                ),
                self._training_kernel_matrix(
                    X_val,
                    Y_val,
                    X_train_mean,
                    Y_train_mean,
                    X_train_std,
                    Y_train_std,
                    N_train
                )
            )
        if return_XTX:
            return self._training_kernel_matrix(
                X_val,
                X_val,
                X_train_mean,
                X_train_mean,
                X_train_std,
                X_train_std,
                N_train
            )
        return self._training_kernel_matrix(
            X_val,
            Y_val,
            X_train_mean,
            Y_train_mean,
            X_train_std,
            Y_train_std,
            N_train
        )

    def _training_kernel_matrix(
            self,
            X_val: np.ndarray,
            mat2_val: np.ndarray,
            X_train_mean: Union[None, np.ndarray] = None,
            mat2_train_mean: Union[None, np.ndarray] = None,
            X_train_std: Union[None, np.ndarray] = None,
            mat2_train_std: Union[None, np.ndarray] = None,
            N_train: Union[None, int] = None,
    ) -> np.ndarray:
        """
        Computes the training set kernel matrix for a given fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.
        
        mat2_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of predictor or resoponse variables.

        X_train_mean : None or array of shape (1, K), optional, default=None
            The row of column-wise means of the training set of predictor variables.

        mat2_train_mean : None or array of shape (1, K) or (1, M), optional,
        default=None
            The row of column-wise means of the training set of predictor or response
            variables.

        X_train_std : None or array of shape (1, K), optional, default=None
            The row of column-wise standard deviations of the training set of predictor
            variables.

        mat2_train_std : None or array of shape (1, K) or (1, M), optional, default=None
            The row of column-wise standard deviations of the training set of predictor
            or response variables.

        N_train : None or int, optional, default=None
            The size of the training set. Only required if `X_train_mean` or
            `mat2_train_mean` is not `None`.

        Returns
        -------
        Array of shape (K, K) or (K, M)
            The training set kernel matrix.
        """
        XTmat2_train = X_val.T @ mat2_val
        if X_train_mean is not None:
            XTmat2_train -= N_train * (X_train_mean.T @ mat2_train_mean)
        if X_train_std is not None and mat2_train_std is not None:
            return XTmat2_train / (X_train_std.T @ mat2_train_std)
        if X_train_std is not None:
            return XTmat2_train / X_train_std.T
        if mat2_train_std is not None:
            return XTmat2_train / mat2_train_std
        return XTmat2_train

    def _compute_training_mat_mean(
            self,
            mat_val: np.ndarray,
            mat_total_mean: np.ndarray,
            N_total_over_N_train: float,
            N_val_over_N_train: float
    ) -> np.ndarray:
        """
        Computes the row of column-wise means of a matrix for a given fold.

        Parameters
        ----------
        mat_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of `X` or `Y`.
        
        mat_total_mean : Array of shape (1, K) or (1, M)
            The row of column-wise means of the total matrix.
        
        N_total_over_N_train : float
            The ratio of the total number of samples to the number of samples in the
            training set.
        
        N_val_over_N_train : float
            The ratio of the number of samples in the validation set to the number of
            samples in the training set.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise means of the training set matrix.
        """
        return (
            N_total_over_N_train * mat_total_mean
            - N_val_over_N_train * np.mean(mat_val, axis=0, keepdims=True)
        )

    def _compute_training_mat_std(
            self,
            mat_val: np.ndarray,
            mat_train_mean: np.ndarray,
            sum_mat_total: np.ndarray,
            sum_sq_mat_total: np.ndarray,
            N_train: int
    ) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of a matrix for a given
        fold.

        Parameters
        ----------
        mat_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of `X` or `Y`.

        mat_train_mean : Array of shape (1, K) or (1, M)
            The row of column-wise means of the training matrix.

        sum_mat_total : Array of shape (1, K) or (1, M)
            The row of column-wise sums of the total matrix.

        sum_sq_mat_total : Array of shape (1, K) or (1, M)
            The row of column-wise sums of squares of the total matrix.

        N_train : int
            The size of the training set.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise standard deviations of the training set matrix.
        """
        train_sum_mat = sum_mat_total - np.expand_dims(
            np.einsum("ij->j", mat_val), axis=0
        )
        train_sum_sq_mat = sum_sq_mat_total - np.expand_dims(
            np.einsum("ij,ij->j", mat_val, mat_val), axis=0
        )
        mat_train_std = np.sqrt(
            1
            / (N_train - 1)
            * (
                -2 * mat_train_mean * train_sum_mat
                + N_train
                * np.einsum("ij,ij -> ij", mat_train_mean, mat_train_mean)
                + train_sum_sq_mat
            )
        )
        mat_train_std[mat_train_std == 0] = 1
        return mat_train_std

    def _init_mat(self, mat: np.ndarray) -> np.ndarray:
        """
        Casts the matrix to the dtype specified in the constructor and reshapes it if
        the matrix is one-dimensional.

        Parameters
        ----------
        mat : Array of shape (N, K) or (N, M) or (N,)
            The matrix to be initialized.

        Returns
        -------
        Array of shape (N, K) or (N, M) or (N, 1)
            The initialized matrix.
        """
        mat = np.asarray(mat, dtype=self.dtype)
        if self.copy and mat.dtype == self.dtype:
            mat = mat.copy()
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        return mat

    def _init_total_stats(self) -> None:
        """
        Initializes the global statistics for `X` and `Y`.
        """
        if self.center_X or self.center_Y or self.scale_X:
            self.X_total_mean = np.mean(self.X_total, axis=0, keepdims=True)
        if (
            (self.center_X or self.center_Y or self.scale_Y)
            and self.Y_total is not None
        ):
            self.Y_total_mean = np.mean(self.Y_total, axis=0, keepdims=True)
        if self.scale_X:
            self.sum_X_total = np.expand_dims(np.einsum("ij->j", self.X_total), axis=0)
            self.sum_sq_X_total = np.expand_dims(
                np.einsum("ij,ij->j", self.X_total, self.X_total), axis=0
            )
        if self.scale_Y and self.Y_total is not None:
            self.sum_Y_total = np.expand_dims(np.einsum("ij->j", self.Y_total), axis=0)
            self.sum_sq_Y_total = np.expand_dims(
                np.einsum("ij,ij->j", self.Y_total, self.Y_total), axis=0
            )

    def _init_val_indices_dict(
        self, cv_splits: Iterable[Hashable]
    ) -> dict[Hashable, npt.NDArray[np.int_]]:
        """
        Generates a list of validation indices for each fold in `cv_splits`.

        Parameters
        ----------
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.
        """
        val_index_dict = {}
        for i, num in enumerate(cv_splits):
            try:
                val_index_dict[num].append(i)
            except KeyError:
                val_index_dict[num] = [i]
        for key in val_index_dict:
            val_index_dict[key] = np.asarray(val_index_dict[key], dtype=int)
        self.val_index_dict = val_index_dict
