"""
Contains the Partitioner class which implements a dictionary to quickly retrieve an
integer index array for validation indices of a given cross-validation partition
(fold). This is an implementation of Algorithm 1 in the paper by O.-C. G. Engstrøm and
M. H. Jensen:
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008

The implementation is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from collections import defaultdict
from collections.abc import Hashable
from typing import Iterable

import numpy as np
import numpy.typing as npt


class Partitioner:
    """
    Implements Algorithm 1 by O.-C. G. Engstrøm and M. H. Jensen:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008
    This class is used to partition data into validation sets based on cross-validation
    folds. It is detached from the CVMatrix so that it does not need to be pickled when
    using the CVMatrix in a multiprocessing context. See the parallel implementation in
    the `ikpls` package by O.-C. G. Engstrøm et al.: https://github.com/sm00thix/ikpls.
    CVMatrix and Partitioner are used together in the cross_validate method
    of the ikpls.fast_cross_validation.numpy_ikpls.PLS class.

    Parameters
    ----------
    folds : Iterable of Hashable with N elements
        An iterable defining cross-validation splits. Each unique value in
        `folds` corresponds to a different fold. The indices of the samples in each fold
        will be stored in a dictionary for quick access.

    Attributes
    ----------
    folds_dict : dict[Hashable, npt.NDArray[np.int_]]
        A dictionary where keys are fold identifiers (from the `folds` parameter) and
        values are NumPy arrays containing the indices of the samples in that fold.
        This allows for efficient retrieval of validation indices for each fold.
    """

    def __init__(self, folds: Iterable[Hashable]) -> None:
        """
        Initializes the Partitioner with a dictionary of folds.

        Parameters:
        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold. The indices of the samples in each
            fold will be stored in `folds_dict` for quick access.
        """
        self.folds_dict: dict[Hashable, npt.NDArray[np.int_]] = {}
        self._init_folds_dict(folds)

    def get_validation_indices(self, fold: Hashable) -> npt.NDArray[np.int_]:
        """
        Returns an integer array of indices of the validation partition samples for
        `fold`.

        Parameters
        ----------
        fold : Hashable
            The fold for which to return the validation partition indices.

        Returns
        -------
        Array of shape (N_val,)
            Integer array of indices of the validation partition samples for the given
            fold.

        Raises
        ------
        ValueError
            If `fold` was not one of the values in the `folds` parameter of the
            constructor.
        """
        try:
            val_indices = self.folds_dict[fold]
        except KeyError as e:
            raise ValueError(f"Fold {fold} not found.") from e
        return val_indices

    def _init_folds_dict(self, folds: Iterable[Hashable]) -> None:
        """
        Generates a dictionary of indices for each fold. The dictionary is stored in
        the `folds_dict` attribute. The dictionary is used to quickly access the
        indices for each fold.

        Parameters
        ----------
        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.
        """
        cur_folds_dict: defaultdict[Hashable, list[int]] = defaultdict(list)
        for i, num in enumerate(folds):
            cur_folds_dict[num].append(i)
        folds_dict: dict[Hashable, npt.NDArray[np.int_]] = {}
        for key in cur_folds_dict:
            folds_dict[key] = np.asarray(cur_folds_dict[key], dtype=int)
        self.folds_dict = folds_dict
