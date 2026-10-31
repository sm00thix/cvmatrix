"""
Contains the CVMatrix class which implements methods for fast computation of training
set kernel matrices in cross-validation using the fast algorithms described in the
paper by O.-C. G. Engstrøm and M. H. Jensen:
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

The implementation is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt

from .partitioner import Partitioner

# Array/scalar type aliases. NumPy is the only required dependency, so JAX is NEVER
# imported at module load -- ``import cvmatrix`` stays JAX-free even when JAX is
# installed. The aliases are numpy-only by default and are broadened in-place to
# also admit ``jax.Array`` values by ``_enable_jax_typing()``, which runs the first
# time the JAX backend is resolved. Because ``from __future__ import annotations``
# keeps annotations as strings, typeguard resolves them against the *current* module
# globals at call time, so the broadened unions take effect once JAX is in use.
Array = np.ndarray
Scalar = Union[np.floating, float, int]
# Integer index arrays (fold validation indices). Under jax.jit/jax.vmap these are
# abstract tracers, so the alias is broadened alongside ``Array``/``Scalar``.
IndexArray = npt.NDArray[np.int_]
# ``dtype`` specifier: ``npt.DTypeLike`` cleanly admits both numpy float types and
# JAX scalar dtypes (e.g. ``jnp.float64``) without importing JAX.
FloatDType = npt.DTypeLike
# Abstract values produced while a function is traced by jax.jit/jax.vmap. Used to
# skip data-dependent validity raises that cannot run under tracing (the host-side
# caller is expected to validate folds before vmap). Empty until the JAX backend is
# used, so the numpy backend's isinstance checks are always False (numpy values are
# never tracers); eager (concrete) jax values are NOT tracers, so they still
# validate like the numpy backend.
_TRACER_TYPES: tuple = ()


def _enable_jax_typing() -> None:
    """Broaden the array/scalar aliases and tracer types to admit JAX values.

    Called (idempotently) when the JAX backend is resolved. Kept out of module
    import so that ``import cvmatrix`` never imports JAX. Because
    ``from __future__ import annotations`` keeps annotations as strings, typeguard
    resolves annotations against these (now broadened) module globals at call time,
    so ``jax.Array`` values pass the runtime type checks once JAX is in use.
    """
    import jax

    global Array, Scalar, IndexArray, _TRACER_TYPES
    Array = Union[np.ndarray, jax.Array]
    Scalar = Union[np.floating, jax.Array, float, int]
    IndexArray = Union[npt.NDArray[np.int_], jax.Array]
    _TRACER_TYPES = (jax.core.Tracer,)


def _resolve_backend(backend: str):
    """
    Resolve the array-namespace module for a given backend name.

    Parameters
    ----------
    backend : {"numpy", "jax"}
        The array backend to use. ``"numpy"`` uses ``numpy`` (the default and only
        required dependency). ``"jax"`` uses ``jax.numpy`` so that the per-fold training
        matrix computations can run inside ``jax.jit``/``jax.vmap`` on CPU/GPU/TPU; it
        requires the optional ``jax`` dependency (``pip install cvmatrix[jax]``).

    Returns
    -------
    module
        The resolved array namespace (``numpy`` or ``jax.numpy``).

    Raises
    ------
    ValueError
        If ``backend`` is not ``"numpy"`` or ``"jax"``.
    ImportError
        If ``backend == "jax"`` but JAX is not installed.
    """
    if backend == "numpy":
        return np
    if backend == "jax":
        try:
            import jax.numpy as jnp
        except ImportError as e:  # pragma: no cover - exercised only without jax
            raise ImportError(
                "backend='jax' requires the optional JAX dependency. Install it with "
                "`pip install cvmatrix[jax]`."
            ) from e
        # Broaden the array/scalar type aliases to admit jax.Array now that JAX is
        # in use (kept out of module import so importing cvmatrix stays JAX-free).
        _enable_jax_typing()
        return jnp
    raise ValueError(f"Invalid backend: {backend!r}. Must be 'numpy' or 'jax'.")


def _reconstruct(cls: type, backend: str, dtype: FloatDType) -> CVMatrix:
    """
    Creates an empty instance of `cls`, a `CVMatrix` or a subclass, with its array
    backend set up. Used by `CVMatrix.__reduce__` for unpickling: pickle calls it before
    restoring the pickled arrays, so that, with the jax backend and a 64-bit `dtype`,
    JAX's 64-bit mode is enabled before the arrays are rebuilt. Otherwise, float64
    arrays would silently become float32 in a fresh process, e.g., a worker process.

    Parameters
    ----------
    cls : type
        The class of the pickled instance.

    backend : {"numpy", "jax"}
        The array backend of the pickled instance.

    dtype : type[np.floating]
        The data type of the pickled instance.

    Returns
    -------
    CVMatrix
        An instance of `cls` with `backend`, `dtype`, and `xp` set.
    """
    obj = cls.__new__(cls)
    obj.backend = backend
    obj.dtype = dtype
    obj._init_backend()
    return obj


class CVMatrix:
    r"""
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

    backend : {"numpy", "jax"}, optional, default="numpy"
        The array backend used for the computations. `"numpy"` uses NumPy, which is
        the only required dependency. `"jax"` uses `jax.numpy` so that the per-fold
        computations of `training_XTX`, `training_XTY`, `training_XTX_XTY`, and
        `training_statistics` can be traced by `jax.jit` and batched over folds with
        `jax.vmap` on CPU, GPU, or TPU. It requires the optional JAX dependency, which
        is installed with `pip install cvmatrix[jax]`; otherwise, an `ImportError` is
        raised. If `"jax"` is used with a 64-bit `dtype`, JAX's 64-bit mode
        (`jax_enable_x64`) is enabled globally so that `dtype` is honored instead of
        being truncated to 32 bits. If you construct the `CVMatrix` inside a function
        traced by `jax.jit` or `jax.vmap`, enable 64-bit mode before tracing, e.g.,
        with `jax.config.update("jax_enable_x64", True)`, because JAX fixes the input
        dtypes when tracing starts.

    compute_XTX : bool, optional, default=True
        Whether `fit` computes the dataset-wide
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`. Its cost grows with the
        square of the number of predictor variables in `X`, so for wide `X` it can
        dominate `fit`. Set this to `False` if neither `training_XTX` nor
        `training_XTX_XTY` will be called. If `False`, these methods raise a
        `ValueError`.

    compute_XTY : bool, optional, default=True
        Whether `fit` computes the dataset-wide
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` when `Y` is provided. Set
        this to `False` if neither `training_XTY` nor `training_XTX_XTY` will be
        called. If `False`, these methods raise a `ValueError`.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        dtype: FloatDType = np.float64,
        copy: bool = True,
        backend: Literal["numpy", "jax"] = "numpy",
        compute_XTX: bool = True,
        compute_XTY: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
        self.copy = copy
        self.compute_XTX = compute_XTX
        self.compute_XTY = compute_XTY
        self.backend = backend
        self._init_backend()
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

    def __reduce__(self) -> tuple:
        """
        Supports pickling with the standard library, e.g., to send a fitted `CVMatrix`
        to worker processes with `multiprocessing` or
        `concurrent.futures.ProcessPoolExecutor`. The array namespace `xp` is a module,
        which cannot be pickled, so it is left out of the pickled state and resolved
        again from `backend` by `_reconstruct` before the state is restored.
        """
        state = self.__dict__.copy()
        del state["xp"]
        return _reconstruct, (type(self), self.backend, self.dtype), state

    def fit(
        self,
        X: npt.ArrayLike,
        Y: Optional[npt.ArrayLike] = None,
        weights: Optional[npt.ArrayLike] = None,
    ) -> None:
        r"""
        Loads and stores `X`, `Y`, and "weights", for cross-validation. Computes
        dataset-wide :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` if
        `compute_XTX` is `True` and, if `Y` is not `None` and `compute_XTY` is `True`,
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`. If `center_X`,
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

            With `backend="jax"`, `weights` may be traced by `jax.jit` or `jax.vmap`,
            e.g., to vmap over datasets that each have their own weights. Traced
            weights cannot be validated, so the caller must check, on the concrete
            weights before tracing (e.g., with `validate_folds`), that they are
            non-negative and that the training set of each fold has at least one
            non-zero weight if `center_X`, `center_Y`, `scale_X`, or `scale_Y` is
            `True`, and more than `ddof` if `scale_X` or `scale_Y` is `True`.
            Otherwise, the results are silently wrong or NaN. The degenerate-fold
            checks of the training methods are skipped whenever the weights or, with
            weights, the validation indices are traced, as a fold's count of non-zero
            training weights is then unknown while tracing. Without weights, the count
            is the training-set size, which is known while tracing, so these checks
            always run. Concrete weights and validation indices are checked as in
            eager execution, even inside `jax.jit`, e.g., when they are closed over.
            Call `fit` inside the traced function and use the fitted instance only
            there, as its attributes then hold tracers.

        Attributes
        ----------
        X : Array
            The total predictor matrix `X` for the entire dataset.

        Y : Array or None
            The total response matrix `Y` for the entire dataset. If `Y` is `None`, this is
            `None`.

        N : int
            The number of samples in the dataset.

        K : int
            The number of predictor variables in `X`.

        M : int or None
            The number of response variables in `Y`. If `Y` is `None`, this is `None`.

        XTX : Array or None
            The total matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}` for the
            entire dataset. This is computed only if `compute_XTX` is `True`.

        XTY : Array or None
            The total matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` for the
            entire dataset. This is computed only if `Y` is not `None` and
            `compute_XTY` is `True`.

        sum_X : Array or None
            The row of column-wise weighted sums of `X` for the entire dataset. This is
            computed only if `center_X`, `scale_X`, or `center_Y` is `True`. This is the
            row of column-wise sums of :math:`\mathbf{W}\mathbf{X}` if `weights` are
            provided and otherwise the row of column-wise sums of :math:`\mathbf{X}`.

        sum_Y : Array or None
            The row of column-wise weighted sums of `Y` for the entire dataset. This is
            computed only if `center_Y`, `scale_Y`, or `center_X` is `True` and `Y` is not
            `None`. This is the row of column-wise sums of :math:`\mathbf{W}\mathbf{Y}` if
            `weights` are provided and otherwise the row of column-wise sums of
            :math:`\mathbf{Y}`.

        sum_sq_X : Array or None
            The row of column-wise weighted squared sums of `X` for the entire dataset.
            This is computed only if `scale_X` is `True`. This is the
            row of column-wise sums of :math:`\mathbf{W}\mathbf{X}\odot\mathbf{X}` if
            `weights` are provided and otherwise the row of column-wise sums of
            :math:`\mathbf{X}\odot\mathbf{X}`.

        sum_sq_Y : Array or None
            The row of column-wise weighted squared sums of `Y` for the entire dataset.
            This is computed only if `scale_Y` is `True` and `Y` is not `None`. This is the
            row of column-wise sums of :math:`\mathbf{W}\mathbf{Y}\odot\mathbf{Y}` if
            `weights` are provided and otherwise the row of column-wise sums of
            :math:`\mathbf{Y}\odot\mathbf{Y}`.

        sq_X : Array or None
            The total weighted squared predictor matrix `X` for the entire dataset. This is
            :math:`\mathbf{W}\mathbf{X}\odot\mathbf{X}`. This is computed only if
            `scale_X` is `True`.

        sq_Y : Array or None
            The total weighted squared response matrix `Y` for the entire dataset. This is
            :math:`\mathbf{W}\mathbf{Y}\odot\mathbf{Y}`. This is computed only if
            `scale_Y` is `True` and `Y` is not `None`.

        WX : Array or None
            The total weighted predictor matrix `X` for the entire dataset. This is
            :math:`\mathbf{W}\mathbf{X}`.

        WY : Array or None
            The total weighted response matrix `Y` for the entire dataset. This is
            :math:`\mathbf{W}\mathbf{Y}`. This is computed only if `Y` is not `None`.

        weights : Array or None
            The total weights for the entire dataset. This is an array of shape (N, 1). If
            `weights` is `None`, this is `None`.

        sum_w : float or None
            The sum of the weights for the entire dataset. If `weights` is `None`, this is
            `None`.

        num_nonzero_w : int or None
            The number of non-zero weights for the entire dataset. If `weights` is `None`,
            this is `None`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `weights` is provided and contains negative values. This is not checked
            for traced weights (see `weights`). Concrete weights are checked even inside
            `jax.jit`. A previously fitted instance is then left unchanged.
        """
        self._init_mats(X, Y, weights)
        self._init_weighted_mats()
        self._init_matrix_products()
        self._init_stats()

    def training_XTX(
        self, validation_indices: IndexArray
    ) -> Tuple[Array, Tuple[Optional[Array], Optional[Array], None, None]]:
        r"""
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        corresponding to every sample except those at the `validation_indices`. Also
        computes the row of column-wise weighted means for `X` and the row of
        column-wise weighted standard deviations for `X`.

        Parameters
        ----------
        validation_indices : IndexArray
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
            - If `fit` did not compute `XTX` because `compute_XTX` was `False`.
            - If `self.center_X` or `self.scale_X` is `True` but cannot be computed due
                to the training set having no non-zero weights.
            - If `self.scale_X` is `True` and the corresponding weighted standard
                deviations cannot be computed due `self.ddof` being greater or equal to
                the number of non-zero weights in the training set.


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

    def training_XTY(
        self, validation_indices: IndexArray
    ) -> Tuple[
        Array,
        Tuple[
            Optional[Array],
            Optional[Array],
            Optional[Array],
            Optional[Array],
        ],
    ]:
        r"""
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`
        corresponding to every sample except those at the `validation_indices`. Also
        computes the row of column-wise weighted means for `X`, the row of column-wise
        weighted standard deviations for `X`, the row of column-wise weighted means for
        `Y`, and the row of column-wise weighted standard deviations for `Y`. If a
        statistic is not computed, it is `None`.

        Parameters
        ----------
        validation_indices : IndexArray
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
            - If `fit` did not compute `XTY` because `compute_XTY` was `False`.
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

    def training_XTX_XTY(
        self, validation_indices: IndexArray
    ) -> Tuple[
        Tuple[Array, Array],
        Tuple[
            Optional[Array],
            Optional[Array],
            Optional[Array],
            Optional[Array],
        ],
    ]:
        r"""
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` corresponding to every
        sample except those at the `validation_indices`. Also computes the row of
        column-wise weighted means for `X`, the row of column-wise weighted standard
        deviations for `X`, the row of column-wise weighted means for `Y`, and the row
        of column-wise weighted standard deviations for `Y`. If a statistic is not
        computed, it is `None`.


        Parameters
        ----------
        validation_indices : IndexArray
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
            - If `fit` did not compute `XTX` because `compute_XTX` was `False`.
            - If `fit` did not compute `XTY` because `compute_XTY` was `False`.
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

    def training_statistics(
        self, validation_indices: IndexArray
    ) -> Tuple[
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
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
        Tuple of four elements of Optional[Array]
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
        return_X_mean = self.center_X or self.scale_X
        return_Y_mean = (self.center_Y or self.scale_Y) and self.Y is not None
        X_val, _, Y_val, _ = self._get_val_matrices(
            val_indices=validation_indices,
            return_X_val=return_X_mean,
            return_X_val_unweighted=False,
            return_Y_val=return_Y_mean,
            return_Y_val_unweighted=False,
        )
        return self._compute_training_stats(
            val_indices=validation_indices,
            X_val=X_val,
            Y_val=Y_val,
            return_X_mean=return_X_mean,
            return_X_std=self.scale_X,
            return_Y_mean=return_Y_mean,
            return_Y_std=self.scale_Y and self.Y is not None,
        )[:-1]  # Exclude the sum of training weights from the return tuple

    def validate_folds(
        self,
        folds: Union[
            Partitioner, Mapping[Hashable, npt.ArrayLike], Iterable[npt.ArrayLike]
        ],
        N: int,
        weights: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Checks folds and weights on the host before they are traced by `jax.jit` or
        `jax.vmap`. The training methods cannot check traced weights, or traced
        validation indices with weights, so this performs their checks on the concrete
        values instead, using this instance's `center_X`, `center_Y`, `scale_X`,
        `scale_Y`, `ddof`, and `dtype`. The instance does not need to be fitted, and
        the checks run with NumPy for both backends.

        Parameters
        ----------
        folds : Partitioner, mapping, or iterable
            The validation indices of each fold: a
            cvmatrix.partitioner.Partitioner, a mapping from each fold to its
            validation indices, or an iterable of validation index arrays, e.g., the
            2-D array of shape (n_folds, N_val) that is batched over with `jax.vmap`.
            In the last case, the folds are named by their position.

        N : int
            The number of samples in the dataset.

        weights : None or array-like of shape (N,) or (N, 1), optional, default=None
            The concrete weights for each sample, or `None` if no weights are used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            - If `weights` does not have `N` elements or contains negative values
              after the cast to `dtype`.
            - If any fold has validation indices outside 0 to `N` - 1. JAX silently
              clamps such indices under tracing.
            - If `center_X`, `center_Y`, `scale_X`, or `scale_Y` is `True` and the
              training set of any fold has no non-zero weights.
            - If `scale_X` or `scale_Y` is `True` and the training set of any fold
              does not have more than `ddof` non-zero weights.

            Every failing fold is listed in the error message. The training-set checks
            are those that any of the training methods may perform, so a fold can be
            rejected that a particular training method would accept.
        """
        if weights is not None:
            weights = np.asarray(weights, dtype=self.dtype).reshape(-1)
            if weights.shape[0] != N:
                raise ValueError(
                    f"`weights` has {weights.shape[0]} elements, but `N` is {N}."
                )
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative.")
            num_nonzero_w = int(np.count_nonzero(weights))
        needs_stats = self.center_X or self.center_Y or self.scale_X or self.scale_Y
        needs_std = self.scale_X or self.scale_Y
        if isinstance(folds, Partitioner):
            folds = folds.folds_dict
        fold_items = folds.items() if isinstance(folds, Mapping) else enumerate(folds)
        problems = []
        for fold, val_indices in fold_items:
            val_indices = np.asarray(val_indices).reshape(-1)
            if val_indices.size and (val_indices.min() < 0 or val_indices.max() >= N):
                problems.append(
                    f"Fold {fold!r}: the validation indices must be in the range 0 to "
                    f"{N - 1}."
                )
                continue
            if not needs_stats:
                continue
            if weights is None:
                num_nonzero_w_train = N - val_indices.size
            else:
                num_nonzero_w_train = num_nonzero_w - int(
                    np.count_nonzero(weights[val_indices])
                )
            if num_nonzero_w_train < 1:
                problems.append(
                    f"Fold {fold!r}: the number of non-zero weights in the training "
                    f"set must be greater than zero, but it is {num_nonzero_w_train}."
                )
            elif needs_std and num_nonzero_w_train <= self.ddof:
                problems.append(
                    f"Fold {fold!r}: the number of non-zero weights in the training "
                    f"set must be greater than `ddof` ({self.ddof}), but it is "
                    f"{num_nonzero_w_train}."
                )
        if problems:
            raise ValueError("Invalid folds:\n" + "\n".join(problems))

    def _as_scalar(self, x):
        """
        Cast a scalar-valued result to the configured dtype.

        For the numpy backend this returns a NumPy scalar (preserving the historical
        ``np.floating`` return types and byte-identical behavior). For the jax backend
        it returns the value unchanged (already a ``jax.numpy`` value of the right
        dtype), keeping the per-fold path trace-safe.
        """
        if self.backend == "numpy":
            return self.dtype(x)
        return x

    def _get_sum_w_train_and_num_nonzero_w_train(
        self, val_indices: IndexArray
    ) -> Tuple[Scalar, Scalar]:
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
        xp = self.xp
        if self.weights is None:
            sum_w_val = val_indices.size
            sum_w_train = self._as_scalar(self.sum_w - sum_w_val)
            # The training-set size is static even under jax.jit/vmap, as array shapes
            # are, so the checks here and in `_compute_std_divisor` always run.
            num_nonzero_w_train = sum_w_train
        else:
            w_val = self.weights[val_indices]
            sum_w_val = xp.sum(w_val)
            sum_w_train = self._as_scalar(self.sum_w - sum_w_val)
            num_nonzero_w_val = xp.count_nonzero(w_val)
            num_nonzero_w_train = self._as_scalar(
                self.num_nonzero_w - num_nonzero_w_val
            )
            if (
                isinstance(num_nonzero_w_train, _TRACER_TYPES)
                and not isinstance(self.weights, _TRACER_TYPES)
                and not isinstance(val_indices, _TRACER_TYPES)
            ):
                # jax.jit staged the count of concrete weights and indices: count again
                # at trace time, so that the checks here and in `_compute_std_divisor`
                # run as in eager execution. Converting to a Python int keeps the
                # comparisons below from being staged as well.
                import jax

                with jax.ensure_compile_time_eval():
                    num_nonzero_w_train = int(
                        xp.count_nonzero(self.weights)
                        - xp.count_nonzero(self.weights[val_indices])
                    )
        # The count is a tracer only if the weights or, with weights, `val_indices` are
        # traced by jax.jit/vmap. Then the check is deferred to a host-side pre-flight
        # on the concrete weights and indices before tracing.
        if (
            not isinstance(num_nonzero_w_train, _TRACER_TYPES)
            and num_nonzero_w_train == 0
        ):
            raise ValueError(
                "The number of non-zero weights in the training set must be "
                "greater than zero."
            )
        return sum_w_train, num_nonzero_w_train

    def _compute_training_stats(
        self,
        val_indices: IndexArray,
        X_val: Optional[Array],
        Y_val: Optional[Array],
        return_X_mean: bool,
        return_X_std: bool,
        return_Y_mean: bool,
        return_Y_std: bool,
    ) -> Tuple[
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Scalar],
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

        Y_val : None or Array of shape (N_val, M)
            The validation set of weighted response variables. If `None`, no statistics
            for `Y` are computed. Required if `return_Y_mean` or `return_Y_std` is
            `True`.

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
        Tuple of Optional[Array]
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
        # Under jax.jit/vmap, the count is a tracer if the weights or, with weights,
        # `val_indices` are traced; data-dependent validity raises are then deferred to
        # a host-side pre-flight (see `_get_sum_w_train_*` and `_compute_std_divisor`).
        # Without weights, the count is the static training-set size, so it is never a
        # tracer and the checks always run.
        traced = isinstance(num_nonzero_w_train, _TRACER_TYPES)
        if return_X_mean or return_X_std:
            sum_X_val = self.xp.sum(X_val, axis=0, keepdims=True)
            sum_X_train = self._compute_train_mat_sum(sum_X_val, self.sum_X)
            X_train_mean = self._compute_training_mat_mean(
                sum_X_train,
                sum_w_train,
            )
        if return_Y_mean or return_Y_std:
            sum_Y_val = self.xp.sum(Y_val, axis=0, keepdims=True)
            sum_Y_train = self._compute_train_mat_sum(sum_Y_val, self.sum_Y)
            Y_train_mean = self._compute_training_mat_mean(
                sum_Y_train,
                sum_w_train,
            )
        if return_X_std or return_Y_std:
            divisor = self._compute_std_divisor(
                sum_w_train, num_nonzero_w_train, traced
            )
        if return_X_std:
            sum_sq_X_val = self.xp.sum(self.sq_X[val_indices], axis=0, keepdims=True)
            sum_sq_X_train = self._compute_train_mat_sum(sum_sq_X_val, self.sum_sq_X)
            X_train_std = self._compute_training_mat_std(
                sum_sq_X_train,
                X_train_mean,
                sum_X_train,
                sum_w_train,
                divisor,
            )
        if return_Y_std:
            sum_sq_Y_val = self.xp.sum(self.sq_Y[val_indices], axis=0, keepdims=True)
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
        self, return_XTX: bool, return_XTY: bool, val_indices: IndexArray
    ) -> Tuple[
        Union[Array, Tuple[Array, Array]],
        Tuple[
            Optional[Array],
            Optional[Array],
            Optional[Array],
            Optional[Array],
        ],
    ]:
        r"""
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and/or :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}` The training
        set corresponds to every sample except those at the `validation_indices`.

        Parameters
        ----------
        return_XTX : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`.

        return_XTY : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`.

        val_indices : IndexArray
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
            If both `return_XTX` and `return_XTY` are `False`, if `return_XTY` is
            `True` and `Y` is `None`, if `return_XTX` is `True` and `fit` did not
            compute `XTX` because `compute_XTX` was `False`, or if `return_XTY` is
            `True` and `fit` did not compute `XTY` because `compute_XTY` was `False`.
        """
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y is None:
            raise ValueError("Response variables `Y` are not provided.")
        # Check the fitted products rather than the flags, so that the flags cannot
        # disagree with what `fit` actually computed.
        if return_XTX and self.XTX is None:
            raise ValueError(
                "`XTX` was not computed by `fit`. Fit a CVMatrix constructed with "
                "`compute_XTX=True` to use `training_XTX` or `training_XTX_XTY`."
            )
        if return_XTY and self.XTY is None:
            raise ValueError(
                "`XTY` was not computed by `fit`. Fit a CVMatrix constructed with "
                "`compute_XTY=True` to use `training_XTY` or `training_XTX_XTY`."
            )
        # The weighted `Y_val` feeds the weighted mean and standard deviation of `Y`,
        # which centering (`center_X` or `center_Y`) and scaling (`scale_Y`) of `XTY`
        # use.
        return_Y_val = return_XTY and self._needs_Y_stats
        X_val, X_val_unweighted, Y_val, Y_val_unweighted = self._get_val_matrices(
            val_indices=val_indices,
            return_X_val=True,
            return_X_val_unweighted=return_XTX,
            return_Y_val=return_Y_val,
            return_Y_val_unweighted=return_XTY,
        )
        X_train_mean, X_train_std, Y_train_mean, Y_train_std, sum_w_train = (
            self._compute_training_stats(
                val_indices=val_indices,
                X_val=(
                    X_val
                    if self.center_X or self.scale_X or (return_XTY and self.center_Y)
                    else None
                ),
                Y_val=Y_val,
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
        self,
        val_indices: IndexArray,
        *,
        return_X_val: bool,
        return_X_val_unweighted: bool,
        return_Y_val: bool,
        return_Y_val_unweighted: bool,
    ) -> Tuple[
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
    ]:
        """
        Returns the requested validation set matrices for a given fold. Only the
        requested matrices are gathered, as gathering the validation set rows is a
        significant part of the cost of each fold.

        Parameters
        ----------
        val_indices : Array of shape (N_val,)
            The indices of the validation set samples for the given fold.

        return_X_val : bool
            Whether to return the validation set of weighted predictor variables.

        return_X_val_unweighted : bool
            Whether to return the validation set of unweighted predictor variables.

        return_Y_val : bool
            Whether to return the validation set of response variables, weighted if
            `WY` is computed and unweighted otherwise.

        return_Y_val_unweighted : bool
            Whether to return the validation set of unweighted response variables.

        Returns
        -------
        Tuple of four elements of Optional[Array]
            The validation sets of predictor variables `X_val` of shape (N_val, K),
            unweighted predictor variables `X_val_unweighted` of shape (N_val, K),
            response variables `Y_val` of shape (N_val, M), and unweighted response
            variables `Y_val_unweighted` of shape (N_val, M). A matrix that is not
            requested is `None`.
        """
        X_val, X_val_unweighted = self._gather_val_pair(
            val_indices, self.WX, self.X, return_X_val, return_X_val_unweighted
        )
        if self.weights is None or not self._needs_Y_stats:
            Y_maybe_weighted = self.Y
        else:
            Y_maybe_weighted = self.WY
        Y_val, Y_val_unweighted = self._gather_val_pair(
            val_indices, Y_maybe_weighted, self.Y, return_Y_val, return_Y_val_unweighted
        )
        return X_val, X_val_unweighted, Y_val, Y_val_unweighted

    @staticmethod
    def _gather_val_pair(
        val_indices: IndexArray,
        mat: Optional[Array],
        mat_unweighted: Optional[Array],
        return_val: bool,
        return_val_unweighted: bool,
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """
        Gathers the requested validation set rows of a possibly weighted matrix and of
        its unweighted counterpart. If both are requested and they are the same array
        (e.g., if no weights are used), the rows are gathered only once.

        Parameters
        ----------
        val_indices : Array of shape (N_val,)
            The indices of the validation set samples for the given fold.

        mat : None or Array of shape (N, K) or (N, M)
            The possibly weighted matrix, `WX` or `WY`.

        mat_unweighted : None or Array of shape (N, K) or (N, M)
            The unweighted matrix, `X` or `Y`.

        return_val : bool
            Whether to return the validation set rows of `mat`.

        return_val_unweighted : bool
            Whether to return the validation set rows of `mat_unweighted`.

        Returns
        -------
        Tuple of two elements of Optional[Array]
            The validation set rows of `mat` and of `mat_unweighted`. A matrix that is
            not requested is `None`.
        """
        mat_val = mat[val_indices] if return_val else None
        if not return_val_unweighted:
            return mat_val, None
        if mat_val is not None and mat is mat_unweighted:
            return mat_val, mat_val
        return mat_val, mat_unweighted[val_indices]

    def _training_kernel_matrix(
        self,
        total_kernel_mat: Array,
        X_val: Array,
        mat2_val: Array,
        X_train_mean: Optional[Array] = None,
        mat2_train_mean: Optional[Array] = None,
        X_train_std: Optional[Array] = None,
        mat2_train_std: Optional[Array] = None,
        sum_w_train: Optional[Scalar] = None,
        center: bool = False,
    ) -> Array:
        r"""
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
        sum_mat_val: Array,
        sum_mat: Array,
    ) -> Array:
        """
        Computes the row vector of column-wise sums of a matrix for a given fold.
        """
        return sum_mat - sum_mat_val

    def _compute_training_mat_mean(
        self,
        sum_mat_train: Array,
        sum_w_train: Scalar,
    ) -> Array:
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
        self, sum_w_train: Scalar, num_nonzero_w_train: Scalar, traced: bool = False
    ) -> Scalar:
        """
        Computes the divisor for the standard deviation calculation based on the number
        of samples in the training set and the number of non-zero weights.

        Parameters
        ----------
        sum_w_train : float
            The size of the training set.

        num_nonzero_w_train : int
            The number of non-zero weights in the training set.

        traced : bool, default=False
            Whether `num_nonzero_w_train` is a tracer, i.e., only known when the
            computation traced by jax.jit/jax.vmap runs: if the weights or, with
            weights, `val_indices` are traced. When True, the data-dependent
            degenerate-fold check is skipped (deferred to a host-side pre-flight); a
            degenerate fold then gives inf or NaN standard deviations.

        Returns
        -------
        float
            The divisor for the standard deviation calculation.
        """
        # Validate eagerly (numpy, or concrete/eager jax). If the count is traced by
        # jax.jit/vmap (`traced`), the check is deferred to a host-side pre-flight on the
        # concrete indices and weights before tracing.
        if not traced and num_nonzero_w_train <= self.ddof:
            raise ValueError(
                "The number of non-zero weights in the training set must be greater "
                "than `ddof`."
            )
        return (num_nonzero_w_train - self.ddof) * sum_w_train / num_nonzero_w_train

    def _compute_training_mat_std(
        self,
        sum_sq_mat_train: Array,
        mat_train_mean: Array,
        sum_mat_train: Array,
        sum_w_train: Scalar,
        divisor: Scalar,
    ) -> Array:
        r"""
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
        xp = self.xp
        mat_train_var = (
            -2 * mat_train_mean * sum_mat_train
            + sum_w_train * mat_train_mean**2
            + sum_sq_mat_train
        ) / divisor
        # `xp.maximum`/`xp.where` instead of boolean-mask in-place assignment: identical
        # results for numpy and trace-safe for jax (jit/vmap).
        mat_train_var = xp.maximum(mat_train_var, 0)
        mat_train_std = xp.sqrt(mat_train_var)
        mat_train_std = xp.where(mat_train_std <= self.resolution, 1, mat_train_std)
        return mat_train_std

    def _init_backend(self) -> None:
        """
        Resolves the array namespace `xp` from `backend`: numpy (the default) or
        jax.numpy. All array operations are routed through `xp` so that, with the jax
        backend, the per-fold `training_*` methods can be traced by jax.jit/jax.vmap.
        The numpy backend is byte-identical to previous releases. JAX defaults to 32
        bits, so with the jax backend and a 64-bit `dtype`, JAX's 64-bit mode is enabled
        so that `dtype` is honored instead of being silently truncated to float32.
        """
        self.xp = _resolve_backend(self.backend)
        if self.backend == "jax" and np.dtype(self.dtype).itemsize >= 8:
            import jax

            jax.config.update("jax_enable_x64", True)

    def _init_mat(self, mat: npt.ArrayLike) -> Array:
        """
        Casts the matrix to the dtype specified in the constructor and reshapes it if
        the matrix is one-dimensional.

        Parameters
        ----------
        mat : Array-like of shape (N, K) or (N, M) or (N,)
            The matrix to be initialized.

        Returns
        -------
        Array of shape (N, K) or (N, M) or (N, 1)
            The initialized matrix.
        """
        mat = self.xp.asarray(mat, dtype=self.dtype)
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

        Raises
        ------
        ValueError
            If `weights` is provided and contains negative values. The instance is then
            left unchanged.
        """
        X = self._init_mat(X)
        Y = None if Y is None else self._init_mat(Y)
        if weights is not None:
            init_weights = self._init_mat(weights)
            negative = self.xp.any(init_weights < 0)
            if isinstance(negative, _TRACER_TYPES) and not isinstance(
                weights, _TRACER_TYPES
            ):
                # jax.jit staged weights that are possibly concrete, e.g., closed over.
                init_weights, negative = self._init_weights_at_trace_time(weights)
            # `negative` is a tracer only for traced weights, which the caller validates
            # before tracing (see `fit`).
            if not isinstance(negative, _TRACER_TYPES) and bool(negative):
                raise ValueError("Weights must be non-negative.")
            weights = init_weights
        # Assign only after validation, so that a fit that raises leaves a previously
        # fitted instance unchanged.
        self.X = X
        self.N, self.K = X.shape
        self.Y = Y
        self.M = None if Y is None else Y.shape[1]
        self.weights = weights

    def _init_weights_at_trace_time(
        self, weights: npt.ArrayLike
    ) -> Tuple[Array, Array]:
        """
        Initializes the weights like `_init_mat` and checks whether any is negative,
        both under `jax.ensure_compile_time_eval()`. Used when jax.jit has staged the
        weights into tracers although they may be concrete, e.g., when closed over:
        concrete weights then stay concrete and are validated as in eager execution,
        here and per fold. Traced weights, e.g., a list of traced scalars, stay traced,
        and so does the returned check. Only called while jax.jit is tracing, so eager
        execution does not pay for the context manager.

        Parameters
        ----------
        weights : Array-like of shape (N,) or (N, 1)
            Weights for each sample in `X` and `Y`.

        Returns
        -------
        Tuple of two elements.
            The initialized weights of shape (N, 1) and whether any of them is negative.
        """
        import jax

        with jax.ensure_compile_time_eval():
            weights = self._init_mat(weights)
            return weights, self.xp.any(weights < 0)

    @property
    def _needs_Y_stats(self) -> bool:
        r"""
        Whether the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`
        uses the weighted statistics of `Y`: the row of column-wise weighted means for
        centering (`center_X` or `center_Y`) and, additionally, the row of column-wise
        weighted standard deviations for scaling (`scale_Y`). Only then are `WY` (with
        weights) and `sum_Y` computed and the weighted validation set rows of `Y`
        gathered.
        """
        return self.center_X or self.center_Y or self.scale_Y

    def _init_weighted_mats(self):
        """
        Initializes the weighted matrices `WX` and `WY`. Without weights, they are `X`
        and `Y` themselves. With weights, they are computed as the product of `X` and
        `Y` with the `weights`, where `WY` is computed only if `Y` is not `None` and its
        weighted statistics are needed (see `_needs_Y_stats`). If `WY` is not
        computed, it is set to `None` so that no `WY` from a previous fit is kept.
        """
        if self.weights is None:
            self.WX = self.X
            self.WY = self.Y
        else:
            self.WX = self.X * self.weights
            if self.Y is not None and self._needs_Y_stats:
                self.WY = self.Y * self.weights
            else:
                self.WY = None

    def _init_matrix_products(self) -> None:
        r"""
        Initializes the global matrix products `XTX` and `XTY` for the
        entire dataset. These are :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{X}`
        and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{W}\mathbf{Y}`, respectively. A product
        that is not computed, due to `compute_XTX` or `compute_XTY` being `False` or
        `Y` being `None`, is set to `None`.
        """
        self.XTX = self.WX.T @ self.X if self.compute_XTX else None
        if self.compute_XTY and self.Y is not None:
            self.XTY = self.WX.T @ self.Y
        else:
            self.XTY = None

    def _init_stats(self) -> None:
        """
        Initializes the global statistics for `X` and `Y`. The statistics of `Y` and
        the squared matrices that are not computed are set to `None` so that none from
        a previous fit are kept.
        """
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            if self.weights is not None:
                self.sum_w = self.xp.sum(self.weights)
                self.num_nonzero_w = self.xp.count_nonzero(self.weights)
            else:
                self.sum_w = self.N
                self.num_nonzero_w = self.N
        if self.center_X or self.center_Y or self.scale_X:
            self.sum_X = self.xp.sum(self.WX, axis=0, keepdims=True)
        if self._needs_Y_stats and self.Y is not None:
            self.sum_Y = self.xp.sum(self.WY, axis=0, keepdims=True)
        else:
            self.sum_Y = None
        if self.scale_X:
            self.sq_X = self.WX * self.X
            self.sum_sq_X = self.xp.sum(self.sq_X, axis=0, keepdims=True)
        else:
            self.sq_X = None
            self.sum_sq_X = None
        if self.scale_Y and self.Y is not None:
            self.sq_Y = self.WY * self.Y
            self.sum_sq_Y = self.xp.sum(self.sq_Y, axis=0, keepdims=True)
        else:
            self.sq_Y = None
            self.sum_sq_Y = None
