"""
This file demonstrates how to use CVMatrix to compute training set matrices
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` with possible centering and scaling of
`X` and `Y` using training set means and standard deviations.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import numpy as np

from cvmatrix.cvmatrix import CVMatrix

if __name__ == "__main__":
    # Create some example data. X must have shape (N, K) or (N,) and Y must have shape
    # (N, M) or (N,). It follows that the number of samples in X and Y must be equal.
    # Y can be None if only XTWX is needed.
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    weights = np.array([4.2, 13.37, 3.14, 0])

    # The cross-validation folds must be of type Hashable (e.g., int, str, etc.)
    # They must be passed in an iterable of length equal to the number of samples in X
    # (which must in turn be equal to the number of samples in Y, if Y is provided.)
    # The splits are grouped by the values in the iterable. In this case, the first
    # the first samples is in the first fold (fold 0), the second sample is in the
    # second fold (fold "one"), the third and fourth samples are in the third fold
    # (fold 2).
    folds = [0, "one", 2, 2]

    # Create a CVMatrix object with centering and scaling of X and Y. We could have
    # used any of the 16 combinations of centering and scaling. The default is to
    # center and scale both X and Y.
    cvm = CVMatrix(
        folds=folds, center_X=True, center_Y=True, scale_X=True, scale_Y=True
    )

    # Fit the model to the data. This will compute total XTWX and XTWY matrices.
    # It also computes global statistics that will be reused when determining the
    # centering and scaling of the training set. Only statistics that are relevant for
    # the chosen centering and scaling are computed.
    cvm.fit(X, Y, weights)

    # The unique folds and associated indices are stored in the
    # folds_dict
    print(f"Folds: {cvm.folds_dict.keys()}")
    for fold, samples in cvm.folds_dict.items():
        print(f"Fold {fold} samples: {samples}")
    print()

    # Compute the training set matrices for each fold.
    print("Training set matrices using training_XTX_XTY:")
    for fold in cvm.folds_dict:
        # Notice that the samples associated with fold are considered part of the
        # validation set. The training set is then all samples not associated with this
        # fold.
        result = cvm.training_XTX_XTY(fold)
        (XTWX, XTWY), (X_mean, X_std, Y_mean, Y_std) = result
        print(f"Fold {fold}:")
        print(f"Training XTWX:\n{XTWX}")
        print(f"Training XTWY:\n{XTWY}")
        print(f"Training weighted X mean:\n{X_mean}")
        print(f"Training weighted X std:\n{X_std}")
        print(f"Training weighted Y mean:\n{Y_mean}")
        print(f"Training weighted Y std:\n{Y_std}")
        print()

    # We can also get only XTWX or only XTWY. However, if both XTWX and XTWY are needed,
    # it is more efficient to call training_XTX_XTY.
    print("Training set matrices using training_XTX and training_XTY:")
    for fold in cvm.folds_dict:
        result = cvm.training_XTX(fold)
        XTWX, (X_mean, X_std, Y_mean, Y_std) = result
        print(f"Fold {fold}:")
        print(f"Training XTWX:\n{XTWX}")
        print(f"Training weighted X mean:\n{X_mean}")
        print(f"Training weighted X std:\n{X_std}")

        # These two are None as they are not computed when only XTX is requested.
        print(f"Training weighted Y mean:\n{Y_mean}")
        print(f"Training weighted Y std:\n{Y_std}")
        print()
    for fold in cvm.folds_dict:
        result = cvm.training_XTY(fold)
        XTWY, (X_mean, X_std, Y_mean, Y_std) = result
        print(f"Fold {fold}:")
        print(f"Training XTWY:\n{XTWY}")
        print(f"Training weighted X mean:\n{X_mean}")
        print(f"Training weighted X std:\n{X_std}")
        print(f"Training weighted Y mean:\n{Y_mean}")
        print(f"Training weighted Y std:\n{Y_std}")
        print()

    # We can also fit on new X and Y. This will recompute the global statistics and
    # allow us to compute training set matrices for the new data, ensuring that the
    # centering and scaling is done correctly.
    X = np.array([[-1, 2, 3], [-4, 5, 6], [-7, 8, 9], [-10, 11, 12]])

    Y = np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8]])

    print("Fitting on new data:")
    cvm.fit(X, Y)
    for fold in cvm.folds_dict:
        result = cvm.training_XTX_XTY(fold)
        (XTWX, XTWY), (X_mean, X_std, Y_mean, Y_std) = result
        print(f"Fold {fold}:")
        print(f"Training XTWX:\n{XTWX}")
        print(f"Training XTWY:\n{XTWY}")
        print(f"Training weighted X mean:\n{X_mean}")
        print(f"Training weighted X std:\n{X_std}")
        print(f"Training weighted Y mean:\n{Y_mean}")
        print(f"Training weighted Y std:\n{Y_std}")
        print()

    # We can also get the training set statistics without computing the training set
    # matrices. This is useful if we only need the statistics for further processing.
    print("Training set statistics:")
    for fold in cvm.folds_dict:
        result = cvm.training_statistics(fold)
        X_mean, X_std, Y_mean, Y_std = result
        print(f"Fold {fold}:")
        print(f"Training weighted X mean:\n{X_mean}")
        print(f"Training weighted X std:\n{X_std}")
        print(f"Training weighted Y mean:\n{Y_mean}")
        print(f"Training weighted Y std:\n{Y_std}")
        print()
