"""
Contains the CVMatrix class which implements methods for fast computation of training
set kernel matrices in cross-validation using the fast algorithms described in the
paper by O.-C. G. Engstrøm and M. H. Jensen:
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

The implementation is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy import typing as npt


class CVMatrix:
    """
    Implements the fast cross-validation algorithms for kernel matrix-based models such
    as PCA, PCR, PLS, and OLS. The algorithms are based on Algorithms 2-7 and all the
    extensions in Table 1 in the paper by O.-C. G. Engstrøm and M. H. Jensen:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008.
    This class is designed to be used in conjunction with
    cvmatrix.validation_partitioner.Partitioner which implements Algorithm 1
    from the same paper.

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

    dtype : type[np.floating], optional, default=np.float64
        The data type used for the computations. The default is `np.float64`.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.

    Attributes
    ----------
    X : np.ndarray
        The total predictor matrix `X` for the entire dataset.

    Y : np.ndarray or None
        The total response matrix `Y` for the entire dataset. If `Y` is `None`, this is
        `None`.

    N : int
        The number of samples in the dataset.

    K : int
        The number of predictor variables in `X`.

    M : int or None
        The number of response variables in `Y`. If `Y` is `None`, this is `None`.

    XTX : np.ndarray
        The total matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` for the
        entire dataset.

    XTY : np.ndarray or None
        The total matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` for the
        entire dataset. This is computed only if `Y` is not `None`.

    sum_X : np.ndarray or None
        The row of column-wise weighted sums of `X` for the entire dataset. This is
        computed only if `center_X`, `scale_X`, or `center_Y` is `True`. This is the
        row of column-wise sums of :math:`\mathbf{W}\mathbf{X}` if `weights` are
        provided and otherwise the row of column-wise sums of :math:`\mathbf{X}`.

    sum_Y : np.ndarray or None
        The row of column-wise weighted sums of `Y` for the entire dataset. This is
        computed only if `center_Y`, `scale_Y`, or `center_X` is `True` and `Y` is not
        `None`. This is the row of column-wise sums of :math:`\mathbf{W}\mathbf{Y}` if
        `weights` are provided and otherwise the row of column-wise sums of
        :math:`\mathbf{Y}`.

    sum_sq_X : np.ndarray or None
        The row of column-wise weighted squared sums of `X` for the entire dataset.
        This is computed only if `scale_X` is `True`. This is the
        row of column-wise sums of :math:`\mathbf{W}\mathbf{X}\odot\mathbf{X}` if
        `weights` are provided and otherwise the row of column-wise sums of
        :math:`\mathbf{X}\odot\mathbf{X}`.

    sum_sq_Y : np.ndarray or None
        The row of column-wise weighted squared sums of `Y` for the entire dataset.
        This is computed only if `scale_Y` is `True` and `Y` is not `None`. This is the
        row of column-wise sums of :math:`\mathbf{W}\mathbf{Y}\odot\mathbf{Y}` if
        `weights` are provided and otherwise the row of column-wise sums of
        :math:`\mathbf{Y}\odot\mathbf{Y}`.

    sq_X : np.ndarray or None
        The total weighted squared predictor matrix `X` for the entire dataset. This is
        :math:`\mathbf{W}\mathbf{X}\odot\mathbf{X}`. This is computed only if
        `scale_X` is `True`.

    sq_Y : np.ndarray or None
        The total weighted squared response matrix `Y` for the entire dataset. This is
        :math:`\mathbf{W}\mathbf{Y}\odot\mathbf{Y}`. This is computed only if
        `scale_Y` is `True` and `Y` is not `None`.

    WX : np.ndarray or None
        The total weighted predictor matrix `X` for the entire dataset. This is
        :math:`\mathbf{W}\mathbf{X}`.

    WY : np.ndarray or None
        The total weighted response matrix `Y` for the entire dataset. This is
        :math:`\mathbf{W}\mathbf{Y}`. This is computed only if `Y` is not `None`.

    weights : np.ndarray or None
        The total weights for the entire dataset. This is an array of shape (N, 1). If
        `weights` is `None`, this is `None`.

    sum_w : float or None
        The sum of the weights for the entire dataset. If `weights` is `None`, this is
        `None`.

    num_nonzero_w : int or None
        The number of non-zero weights for the entire dataset. If `weights` is `None`,
        this is `None`.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.dtype = dtype
        self.copy = copy
        self.resolution = np.finfo(dtype).resolution * 10
        self.X = None
        self.Y = None
        self.N = None
        self.K = None
        self.M = None
        self.XTX = None
        self.XTY = None
        self.sum_X = None
        self.sum_Y = None
        self.sum_sq_X = None
        self.sum_sq_Y = None
        self.sq_X = None
        self.sq_Y = None
        self.WX = None
        self.WY = None
        self.weights = None
        self.sum_w = None
        self.num_nonzero_w = None

    def fit(
        self,
        X: npt.ArrayLike,
        Y: Optional[npt.ArrayLike] = None,
        weights: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Loads and stores `X`, `Y`, and "weights", for cross-validation. Computes
        dataset-wide :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and, if `Y` is
        not `None`, :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. If `center_X`,
        `center_Y`, `scale_X`, or `scale_Y` is `True`, the corresponding global
        statistics are also computed.

        Parameters
        ----------
        X : Array-like of shape (N, K) or (N,)
            Predictor variables for the entire dataset.

        Y : None or array-like of shape (N, M) or (N,), optional, default=None
            Response variables for the entire dataset. If `None`, subsequent calls to
            training_XTY and training_XTX_XTY will raise a `ValueError`.

        weights : None or array-like of shape (N,) or (N, 1), optional, default=None
            Weights for each sample in `X` and `Y`. If `None`, no weights are used in
            the computations. If provided, the weights must be non-negative.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `weights` is provided and contains negative values.
        """
        self._init_mats(X, Y, weights)
        self._init_weighted_mats()
        self._init_matrix_products()
        self._init_stats()

    def training_XTX(
        self, validation_indices: npt.NDArray[np.int_]
    ) -> Tuple[
        np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray], None, None]
    ]:
        """
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        corresponding to every sample except those at the `validation_indices`. Also
        computes the row of column-wise weighted means for `X` and the row of
        column-wise weighted standard deviations for `X`.

        Parameters
        ----------
        validation_indices : npt.NDArray[np.int_]
            An integer array of indices for the validation set for which to
            return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`. The validation indices
            should be obtained using the `get_validation_indices` method of a
            cvmatrix.validation_partitioner.Partitioner object.

        Returns
        -------
        Tuple of two elements.
            The first element is an array of shape (K, K) corresponding to the training
            set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`. The second element
            is a tuple containing the row of column-wise weighted means for `X`, the
            row of column-wise weighted standard deviations for `X`, and two `None`
            corresponding to the non-computed rows of column-wise weighted means and
            standard deviations for `Y`. If a statistic is not computed, it is `None`.

        Raises
        ------
        ValueError
            - If `Y` is `None`.
            - If `self.center_X`, `self.center_Y`, `self.scale_X`, or `self.scale_Y` is
                `True` and the corresponding weighted means and standard deviations
                cannot be computed due to the training set having no non-zero
                weights.
            - If `self.scale_X` or `self.scale_Y` is `True` and the corresponding
                weighted standard deviations cannot be computed due `self.ddof` being
                greater or equal to the number of non-zero weights in the training
                set.


        See Also
        --------
        training_XTY :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` and weighted
            statistics.
        training_XTX_XTY :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`,
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`, and weighted
            statistics. This method is faster than calling
            `training_XTX` and `training_XTY` separately.
        """
        return self._training_matrices(True, False, validation_indices)

    def training_XTY(self, validation_indices: npt.NDArray[np.int_]) -> Tuple[
        np.ndarray,
        Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ],
    ]:
        """
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`
        corresponding to every sample except those at the `validation_indices`. Also
        computes the row of column-wise weighted means for `X`, the row of column-wise
        weighted standard deviations for `X`, the row of column-wise weighted means for
        `Y`, and the row of column-wise weighted standard deviations for `Y`. If a
        statistic is not computed, it is `None`.

        Parameters
        ----------
        validation_indices : npt.NDArray[np.int_]
            An integer array of indices for the validation set for which to
            return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The validation indices
            should be obtained using the `get_validation_indices` method of a
            cvmatrix.validation_partitioner.Partitioner object.

        Returns
        -------
        Tuple of two elements.
            The first element is an array of shape (K, M) corresponding to the training
            set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The second element
            is a tuple containing the row of column-wise weighted means for `X`, the
            row of column-wise weighted standard deviations for `X`, the row of
            column-wise weighted means for `Y`, and the row of column-wise weighted
            standard deviations for `Y`. If a statistic is not computed, it is `None`.

        Raises
        ------
        ValueError
            - If `Y` is `None`.
            - If `self.center_X`, `self.center_Y`, `self.scale_X`, or `self.scale_Y` is
                `True` and the corresponding weighted means and standard deviations
                cannot be computed due to the training set having no non-zero
                weights.
            - If `self.scale_X` or `self.scale_Y` is `True` and the corresponding
                weighted standard deviations cannot be computed due `self.ddof` being
                greater or equal to the number of non-zero weights in the training
                set.

        See Also
        --------
        training_XTX :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and weighted
            statistics.
        training_XTX_XTY :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`,
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`, and weighted
            statistics. This method is faster than calling `training_XTX` and
            `training_XTY` separately.
        """
        return self._training_matrices(False, True, validation_indices)

    def training_XTX_XTY(self, validation_indices: npt.NDArray[np.int_]) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ],
    ]:
        """
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` corresponding to every
        sample except those at the `validation_indices`. Also computes the row of
        column-wise weighted means for `X`, the row of column-wise weighted standard
        deviations for `X`, the row of column-wise weighted means for `Y`, and the row
        of column-wise weighted standard deviations for `Y`. If a statistic is not
        computed, it is `None`.


        Parameters
        ----------
        validation_indices : npt.NDArray[np.int_]
            An integer array of indices for the validation set for which to
            return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The validation indices
            should be obtained using the `get_validation_indices` method of a
            cvmatrix.validation_partitioner.Partitioner object.

        Returns
        -------
        Tuple of two tuples.
            The first tuple contains arrays of shapes (K, K) and (K, M). These are the
            training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The second tuple
            contains the row of column-wise weighted means for `X`, the row of
            column-wise weighted standard deviations for `X`, the row of column-wise
            weighted means for `Y`, and the row of column-wise weighted standard
            deviations for `Y`. If a statistic is not computed, it is `None`.

        Raises
        ------
        ValueError
            - If `Y` is `None`.
            - If `self.center_X`, `self.center_Y`, `self.scale_X`, or `self.scale_Y` is
                `True` and the corresponding weighted means and standard deviations
                cannot be computed due to the training set having no non-zero
                weights.
            - If `self.scale_X` or `self.scale_Y` is `True` and the corresponding
                weighted standard deviations cannot be computed due `self.ddof` being
                greater or equal to the number of non-zero weights in the training
                set.

        See Also
        --------
        training_XTX :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and weighted
            statistics.
        training_XTY :
            Computes the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` and weighted
            statistics.
        """
        return self._training_matrices(True, True, validation_indices)

    def training_statistics(self, validation_indices: npt.NDArray[np.int_]) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Computes the row of column-wise weighted means and standard deviations for `X`
        and `Y` corresponding to every sample except those at `validation_indices`. The
        statistics that can be computed depend on the arguments provided in the
        constructor: `X` mean can be computed if `center_X` or `scale_X`, or `center_Y`
        is `True`. `X` standard deviation can be computed if `scale_X` is True. `Y`
        mean can be computed if `center_X ,`center_Y` or `scale_Y` is `True`, and `Y`
        is provided. `Y` standard deviation can be computed if `scale_Y` is `True` and
        `Y` is provided.

        Parameters
        ----------
        fold : Hashable
            The fold for which to return the corresponding training statistics.

        Returns
        -------
        Tuple of four elements of Optional[np.ndarray]
            A tuple containing the row of column-wise weighted means for `X`, the row
            of column-wise weighted standard deviations for `X`, the row of column-wise
            weighted means for `Y`, and the row of column-wise weighted standard
            deviations for `Y`. If a statistic is not computed, it is `None`.

        Raises
        ------
        ValueError
            - If `self.center_X`, `self.center_Y`, `self.scale_X`, or `self.scale_Y` is
                `True` and the corresponding weighted means and standard deviations
                cannot be computed due to the training set having no non-zero
                weights.
            - If `self.scale_X` or `self.scale_Y` is `True` and the corresponding
                weighted standard deviations cannot be computed due `self.ddof` being
                greater or equal to the number of non-zero weights in the training
                set.

        """
        X_val, X_val_unweighted, Y_val, Y_val_unweighted = self._get_val_matrices(
            val_indices=validation_indices, return_XTY=self.Y is not None
        )
        return self._compute_training_stats(
            val_indices=validation_indices,
            X_val=X_val,
            X_val_unweighted=X_val_unweighted,
            Y_val=Y_val,
            Y_val_unweighted=Y_val_unweighted,
            return_X_mean=self.center_X or self.scale_X,
            return_X_std=self.scale_X,
            return_Y_mean=(self.center_Y or self.scale_Y) and self.Y is not None,
            return_Y_std=self.scale_Y and self.Y is not None,
        )[
            :-1
        ]  # Exclude the sum of training weights from the return tuple

    def _get_sum_w_train_and_num_nonzero_w_train(
        self, val_indices: npt.NDArray[np.int_]
    ) -> Tuple[float, float]:
        """
        Returns a tuple containing the sum of weights in the training set and the number
        of non-zero weights in the training set. If `self.weights` is `None`, it returns
        the size of the training set as both the sum of weights and the number of
        non-zero weights.

        Returns
        -------
        Tuple of floats
            The sum of weights in the training set and the number of non-zero weights in
            the training set.

        Raises
        ------
        ValueError
            If the number of non-zero weights in the training set is zero, which would
            make it impossible to compute either of training set means or standard
            deviations.
        """
        if self.weights is None:
            sum_w_val = np.asarray(val_indices.size, dtype=self.dtype)
            sum_w_train = self.sum_w - sum_w_val
            return (sum_w_train, sum_w_train)
        w_val = self.weights[val_indices]
        sum_w_val = np.sum(w_val)
        sum_w_train = self.sum_w - sum_w_val
        num_nonzero_w_val = np.count_nonzero(w_val)
        num_nonzero_w_train = np.asarray(
            self.num_nonzero_w - num_nonzero_w_val, dtype=self.dtype
        )
        if num_nonzero_w_train == 0:
            raise ValueError(
                "The number of non-zero weights in the training set must be "
                "greater than zero."
            )
        return sum_w_train, num_nonzero_w_train

    def _compute_training_stats(
        self,
        val_indices: npt.NDArray[np.int_],
        X_val: Optional[np.ndarray],
        X_val_unweighted: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        Y_val_unweighted: Optional[np.ndarray],
        return_X_mean: bool,
        return_X_std: bool,
        return_Y_mean: bool,
        return_Y_std: bool,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
    ]:
        """
        Computes the training set statistics. The training set corresponds
        to every sample except those at the `validation_indices`. The statistics
        include the row of column-wise weighted means and standard deviations for `X`
        and `Y`.

        Parameters
        ----------
        val_indices : Array of shape (N_val,)
            The indices of the validation set samples for which to compute
            statistics on the corresponding training set.

        X_val : None or Array of shape (N_val, K)
            The validation set of weighted predictor variables. If `None`, no
            statistics for `X` are computed. Required if `return_X_mean` or
            `return_X_std` is `True`.

        X_val_unweighted : None or Array of shape (N_val, K)
            The validation set of unweighted predictor variables. Required if
            `return_X_std` is `True`.

        Y_val : None or Array of shape (N_val, M)
            The validation set of weighted response variables. If `None`, no statistics
            for `Y` are computed. Required if `return_Y_mean` or `return_Y_std` is
            `True`.

        Y_val_unweighted : None or Array of shape (N_val, M)
            The validation set of unweighted response variables. Required if
            `return_Y_std` is `True`.

        return_X_mean : bool
            Whether to compute the row of column-wise weighted means for `X`.

        return_X_std : bool
            Whether to compute the row of column-wise weighted standard deviations for
            `X`.

        return_Y_mean : bool
            Whether to compute the row of column-wise weighted means for `Y`.

        return_Y_std : bool
            Whether to compute the row of column-wise weighted standard deviations for
            `Y`.

        Returns
        -------
        Tuple of Optional[np.ndarray]
            A tuple containing the row of column-wise weighted means for `X`, the row
            of column-wise weighted standard deviations for `X`, the row of column-wise
            weighted means for `Y`, the row of column-wise weighted standard deviations
            for `Y`, and the sum of training weights. If a statistic is not computed,
            it is `None`.
        """
        if (
            not return_X_mean
            and not return_X_std
            and not return_Y_mean
            and not return_Y_std
        ):
            return None, None, None, None, None
        sum_w_train, num_nonzero_w_train = (
            self._get_sum_w_train_and_num_nonzero_w_train(val_indices)
        )
        if return_X_mean or return_X_std:
            sum_X_val = np.sum(X_val, axis=0, keepdims=True)
            sum_X_train = self._compute_train_mat_sum(sum_X_val, self.sum_X)
            X_train_mean = self._compute_training_mat_mean(
                sum_X_train,
                sum_w_train,
            )
        if return_Y_mean or return_Y_std:
            sum_Y_val = np.sum(Y_val, axis=0, keepdims=True)
            sum_Y_train = self._compute_train_mat_sum(sum_Y_val, self.sum_Y)
            Y_train_mean = self._compute_training_mat_mean(
                sum_Y_train,
                sum_w_train,
            )
        if return_X_std or return_Y_std:
            divisor = self._compute_std_divisor(sum_w_train, num_nonzero_w_train)
        if return_X_std:
            sum_sq_X_val = np.sum(self.sq_X[val_indices], axis=0, keepdims=True)
            sum_sq_X_train = self._compute_train_mat_sum(sum_sq_X_val, self.sum_sq_X)
            X_train_std = self._compute_training_mat_std(
                sum_sq_X_train,
                X_train_mean,
                sum_X_train,
                sum_w_train,
                divisor,
            )
        if return_Y_std:
            sum_sq_Y_val = np.sum(self.sq_Y[val_indices], axis=0, keepdims=True)
            sum_sq_Y_train = self._compute_train_mat_sum(sum_sq_Y_val, self.sum_sq_Y)
            Y_train_std = self._compute_training_mat_std(
                sum_sq_Y_train,
                Y_train_mean,
                sum_Y_train,
                sum_w_train,
                divisor,
            )
        return (
            X_train_mean if return_X_mean else None,
            X_train_std if return_X_std else None,
            Y_train_mean if return_Y_mean else None,
            Y_train_std if return_Y_std else None,
            sum_w_train,
        )

    def _training_matrices(
        self, return_XTX: bool, return_XTY: bool, val_indices: npt.NDArray[np.int_]
    ) -> Tuple[
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ],
    ]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and/or :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` The training
        set corresponds to every sample except those at the `validation_indices`.

        Parameters
        ----------
        return_XTX : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`.

        return_XTY : bool, optional, default=False
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`.

        validation_indices : npt.NDArray[np.int_]
            An integer array of indices for the validation set for which to
            return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The validation indices
            should be obtained using the `get_validation_indices` method of a
            cvmatrix.validation_partitioner.Partitioner object.

        Returns
        -------
        Tuple of two elements.
            The first element is an array of shape (K, K) or (K, M) or a tuple of
            arrays of shapes (K, K) and (K, M). These are the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` and/or training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. The second element is
            a tuple containing the row of column-wise weighted means for `X`, the row
            of column-wise weighted standard deviations for `X`, the row of column-wise
            weighted means for `Y`, and the row of column-wise weighted standard
            deviations for `Y`. If a statistic is not computed, it is `None`.

        Raises
        ------
        ValueError
            If both `return_XTX` and `return_XTY` are `False` or if `return_XTY` is
            `True` and `Y` is `None`.
        """
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y is None:
            raise ValueError("Response variables `Y` are not provided.")
        X_val, X_val_unweighted, Y_val, Y_val_unweighted = self._get_val_matrices(
            val_indices=val_indices, return_XTY=return_XTY
        )
        X_train_mean, X_train_std, Y_train_mean, Y_train_std, sum_w_train = (
            self._compute_training_stats(
                val_indices=val_indices,
                X_val=(
                    X_val
                    if self.center_X or self.scale_X or (return_XTY and self.center_Y)
                    else None
                ),
                X_val_unweighted=X_val_unweighted if self.scale_X else None,
                Y_val=(
                    Y_val
                    if return_XTY and (self.center_X or self.center_Y or self.scale_Y)
                    else None
                ),
                Y_val_unweighted=(
                    Y_val_unweighted if return_XTY and self.scale_Y else None
                ),
                return_X_mean=self.center_X or (return_XTY and self.center_Y),
                return_Y_mean=return_XTY and (self.center_X or self.center_Y),
                return_X_std=self.scale_X,
                return_Y_std=return_XTY and self.scale_Y,
            )
        )
        stats_tuple = (
            X_train_mean,
            X_train_std,
            Y_train_mean,
            Y_train_std,
        )
        if return_XTX and return_XTY:
            return (
                (
                    self._training_kernel_matrix(
                        self.XTX,
                        X_val,
                        X_val_unweighted,
                        X_train_mean,
                        X_train_mean,
                        X_train_std,
                        X_train_std,
                        sum_w_train,
                        center=self.center_X,
                    ),
                    self._training_kernel_matrix(
                        self.XTY,
                        X_val,
                        Y_val_unweighted,
                        X_train_mean,
                        Y_train_mean,
                        X_train_std,
                        Y_train_std,
                        sum_w_train,
                        center=self.center_X or self.center_Y,
                    ),
                ),
                stats_tuple,
            )
        if return_XTX:
            return (
                self._training_kernel_matrix(
                    self.XTX,
                    X_val,
                    X_val_unweighted,
                    X_train_mean,
                    X_train_mean,
                    X_train_std,
                    X_train_std,
                    sum_w_train,
                    center=self.center_X,
                ),
                stats_tuple,
            )
        return (
            self._training_kernel_matrix(
                self.XTY,
                X_val,
                Y_val_unweighted,
                X_train_mean,
                Y_train_mean,
                X_train_std,
                Y_train_std,
                sum_w_train,
                center=self.center_X or self.center_Y,
            ),
            stats_tuple,
        )

    def _get_val_matrices(
        self, val_indices: npt.NDArray[np.int_], return_XTY: bool
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Returns the validation set matrices for a given fold.
        Parameters
        ----------
        val_indices : Array of shape (N_val,)
            The indices of the validation set samples for the given fold.
        return_XTY : bool
            Whether to return the validation set of response variables `Y`. If `False`,
            the returned `Y_val` and `Y_val_unweighted` will be `None`.
        Returns
        -------
        Tuple of arrays of shapes (N_val, K), (N_val, K), (N_val, M), and (N_val, M)
            The validation set of predictor variables `X`, the validation set of
            unweighted predictor variables `X_unweighted`, the validation set of
            response variables `Y`, and the validation set of unweighted response
            variables `Y_unweighted`. If `return_XTY` is `False`, `Y` and
            `Y_unweighted` will be `None`.
        """
        X_val = self.WX[val_indices]
        if self.weights is None:
            X_val_unweighted = X_val
        else:
            X_val_unweighted = self.X[val_indices]
        if return_XTY:
            if self.weights is None or not (
                self.center_X or self.center_Y or self.scale_Y
            ):
                Y_val = self.Y[val_indices]
                Y_val_unweighted = Y_val
            else:
                Y_val = self.WY[val_indices]
                Y_val_unweighted = self.Y[val_indices]
        else:
            Y_val = None
            Y_val_unweighted = None
        return X_val, X_val_unweighted, Y_val, Y_val_unweighted

    def _training_kernel_matrix(
        self,
        total_kernel_mat: np.ndarray,
        X_val: np.ndarray,
        mat2_val: np.ndarray,
        X_train_mean: Optional[np.ndarray] = None,
        mat2_train_mean: Optional[np.ndarray] = None,
        X_train_std: Optional[np.ndarray] = None,
        mat2_train_std: Optional[np.ndarray] = None,
        sum_w_train: Optional[float] = None,
        center: bool = False,
    ) -> np.ndarray:
        """
        Computes the training set kernel matrix for a given fold.

        Parameters
        ----------
        total_kernel_mat : Array of shape (N, K) or (N, M)
            The total kernel matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
            or :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`.

        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.

        mat2_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of predictor or resoponse variables.

        X_train_mean : None or array of shape (1, K), optional, default=None
            The row of column-wise weighted means of the training set of predictor
            variables.

        mat2_train_mean : None or array of shape (1, K) or (1, M), optional,
        default=None
            The row of column-wise weighted means of the training set of predictor or
            response variables.

        X_train_std : None or array of shape (1, K), optional, default=None
            The row of column-wise weighted standard deviations of the training set of
            predictor variables.

        mat2_train_std : None or array of shape (1, K) or (1, M), optional, default=None
            The row of column-wise weighted standard deviations of the training set of
            predictor or response variables.

        sum_w_train : None or float, optional, default=None
            The sum of weights in the training set. Only required if `X_train_mean` or
            `mat2_train_mean` is not `None`.

        center : bool, optional, default=False
            Whether to center the kernel matrix. If `True`, the kernel matrix is
            centered. Setting this parameter to `True` requires that `X_train_mean` and
            `mat2_train_mean` are not `None`.

        Returns
        -------
        Array of shape (K, K) or (K, M)
            The training set kernel matrix.
        """
        XTmat2_train = total_kernel_mat - X_val.T @ mat2_val
        if center:
            XTmat2_train -= sum_w_train * (X_train_mean.T @ mat2_train_mean)
        if X_train_std is not None and mat2_train_std is not None:
            return XTmat2_train / (X_train_std.T @ mat2_train_std)
        if X_train_std is not None:
            return XTmat2_train / X_train_std.T
        if mat2_train_std is not None:
            return XTmat2_train / mat2_train_std
        return XTmat2_train

    def _compute_train_mat_sum(
        self,
        sum_mat_val: np.ndarray,
        sum_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the row vector of column-wise sums of a matrix for a given fold.
        """
        return sum_mat - sum_mat_val

    def _compute_training_mat_mean(
        self,
        sum_mat_train: np.ndarray,
        sum_w_train: float,
    ) -> np.ndarray:
        """
        Computes the row of column-wise means of a matrix for a given fold.

        Parameters
        ----------
        sum_mat_train : Array of shape (1, K) or (1, M)
            The row of column-wise sums of the training set of `WX` or `WY`.

        sum_w_train : float
            The sum of weights in the training set.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise means of the training set matrix.
        """
        return sum_mat_train / sum_w_train

    def _compute_std_divisor(
        self, sum_w_train: float, num_nonzero_w_train: int
    ) -> float:
        """
        Computes the divisor for the standard deviation calculation based on the number
        of samples in the training set and the number of non-zero weights.

        Parameters
        ----------
        sum_w_train : float
            The size of the training set.

        num_nonzero_w_train : int
            The number of non-zero weights in the training set.

        Returns
        -------
        float
            The divisor for the standard deviation calculation.
        """
        if num_nonzero_w_train <= self.ddof:
            raise ValueError(
                "The number of non-zero weights in the training set must be greater "
                "than `ddof`."
            )
        return (num_nonzero_w_train - self.ddof) * sum_w_train / num_nonzero_w_train

    def _compute_training_mat_std(
        self,
        sum_sq_mat_train: np.ndarray,
        mat_train_mean: np.ndarray,
        sum_mat_train: np.ndarray,
        sum_w_train: float,
        divisor: float,
    ) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of a matrix for a given
        fold.

        Parameters
        ----------
        sum_sq_mat_train : Array of shape (1, K) or (1, M)
            The row of column-wise sums of products between the total weighted matrix
            and the total unweighted matrix. This is :math:`\mathbf{W}\mathbf{X}\odot\mathbf{X}`
            or :math:`\mathbf{W}\mathbf{Y}\odot\mathbf{Y}`.

        mat_train_mean : Array of shape (1, K) or (1, M)
            The row of column-wise weighted means of the training matrix.

        sum_mat_train : Array of shape (1, K) or (1, M)
            The row of column-wise sums of the training set of `WX` or `WY`.

        sum_w_train : float
            The size of the training set.

        divisor : float
            The divisor for the standard deviation calculation. Computed using
            `_compute_std_divisor`.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise standard deviations of the training set matrix.
        """
        mat_train_var = (
            -2 * mat_train_mean * sum_mat_train
            + sum_w_train * mat_train_mean**2
            + sum_sq_mat_train
        ) / divisor
        mat_train_var[mat_train_var < 0] = 0
        mat_train_std = np.sqrt(mat_train_var)
        mat_train_std[np.abs(mat_train_std) <= self.resolution] = 1
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

    def _init_mats(
        self,
        X: npt.ArrayLike,
        Y: Optional[npt.ArrayLike],
        weights: Optional[npt.ArrayLike],
    ) -> None:
        """
        Initializes the matrices `X`, `Y`, and `weights` with the provided
        data. If `Y` is `None`, then `Y` is not initialized. If `weights` is
        provided, it initializes the weighted matrices `WX` and `WY`.

        Parameters
        ----------
        X : Array-like of shape (N, K) or (N,)
            Predictor variables for the entire dataset.

        Y : None or array-like of shape (N, M) or (N,), optional, default=None
            Response variables for the entire dataset. If `None`, subsequent calls to
            training_XTY and training_XTX_XTY will raise a `ValueError`.

        weights : None or array-like of shape (N,) or (N, 1), optional, default=None
            Weights for each sample in `X` and `Y`. If `None`, no weights are used in
            the computations.
        """
        self.X = self._init_mat(X)
        self.N, self.K = self.X.shape
        if Y is not None:
            self.Y = self._init_mat(Y)
            self.M = self.Y.shape[1]
        else:
            self.Y = None
            self.M = None

        if weights is not None:
            self.weights = self._init_mat(weights)
            if np.any(self.weights < 0):
                raise ValueError("Weights must be non-negative.")
        else:
            self.weights = None

    def _init_weighted_mats(self):
        """
        Initializes the weighted matrices `WX` and `WY` if weights are
        provided. These matrices are computed as the product of the original matrices
        `X` and `Y` with the `weights`. If `Y` is `None`, then `WY` is not initialized.
        If `w` is `None`, then this method does nothing.
        """
        if self.weights is None:
            self.WX = self.X
            if self.Y is not None:
                self.WY = self.Y
        else:
            self.WX = self.X * self.weights
            if self.Y is not None and (self.center_X or self.center_Y or self.scale_Y):
                self.WY = self.Y * self.weights

    def _init_matrix_products(self) -> None:
        """
        Initializes the global matrix products `XTX` and `XTY` for the
        entire dataset. These are :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`, respectively.
        """
        self.XTX = self.WX.T @ self.X
        if self.Y is not None:
            self.XTY = self.WX.T @ self.Y

    def _init_stats(self) -> None:
        """
        Initializes the global statistics for `X` and `Y`.
        """
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            if self.weights is not None:
                self.sum_w = np.sum(self.weights)
                self.num_nonzero_w = np.count_nonzero(self.weights)
            else:
                self.sum_w = self.N
                self.num_nonzero_w = self.N
        if self.center_X or self.center_Y or self.scale_X:
            self.sum_X = np.sum(self.WX, axis=0, keepdims=True)
        if (self.center_X or self.center_Y or self.scale_Y) and self.Y is not None:
            self.sum_Y = np.sum(self.WY, axis=0, keepdims=True)
        if self.scale_X:
            self.sq_X = self.WX * self.X
            self.sum_sq_X = np.sum(self.sq_X, axis=0, keepdims=True)
        else:
            self.sum_sq_X = None
        if self.scale_Y and self.Y is not None:
            self.sq_Y = self.WY * self.Y
            self.sum_sq_Y = np.sum(self.sq_Y, axis=0, keepdims=True)
        else:
            self.sum_sq_Y = None
