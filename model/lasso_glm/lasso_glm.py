#!/usr/bin/env python3
import os
import copy as cp
import numpy as np
import tensorflow as tf
import matlab.engine
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))

__all__ = [
    "lasso_glm",
]

class lasso_glm:
    """
    `lasso_glm` classifier, with each classifier for each time point.
    """

    def __init__(self, params):
        """
        Initialize `lasso_glm` object.

        Args:
            params: Model parameters initialized by lasso_glm_params.

        Returns:
            None
        """
        # First call super class init function to set up `object`
        # style model and inherit it's functionality.
        super(lasso_glm, self).__init__()

        # Copy hyperparameters (e.g. initialized parameters) from parameter dotdict, usually
        # generated from lasso_glm_params() in params/lasso_glm_params.py.
        self.params = cp.deepcopy(params)

    # def fit func
    def fit(self, X, y):
        """
        Fit matlab `lasso_glm` model with (X_train, y_train), then each category will leave at least
        `n_lxo` samples to test the performance of `lasso_glm` model.

        Args:
            X: (3[tuple],) - The input data (X_train, X_test1, X_test2),
                each item is of shape (n_samples, seq_len, n_channels).
            y: (3[tuple],) - The target labels (y_train, y_test1, y_test2), each item is of shape (n_samples,).

        Returns:
            acc_test1: (seq_len,) - The test1 accuracy of ecah asso classifier trained at each time point.
            acc_test2: (seq_len,) - The test accuracy of ecah asso classifier trained at each time point.
        """
        # Initialize `X_train` & `X_test1` & `X_test2` from `X`, `y_train` & `y_test1` & `y_test2` from `y`.
        X_train, X_test1, X_test2 = X; y_train, y_test1, y_test2 = y
        # Check whether `X` & `y` are well-structured.
        y_train = np.expand_dims(y_train, axis=-1)
        assert (len(X_train.shape) == 3) and (len(y_train.shape) == 2)
        y_test1 = np.expand_dims(y_test1, axis=-1)
        assert (len(X_test1.shape) == 3) and (len(y_test1.shape) == 2)
        y_test2 = np.expand_dims(y_test2, axis=-1)
        assert (len(X_test2.shape) == 3) and (len(y_test2.shape) == 2)
        # Start matlab engine.
        mat_eng = matlab.engine.start_matlab()
        # Add path of current directory.
        mat_eng.addpath(os.path.dirname(os.path.abspath(__file__)))
        # Prepare input for calling matlab function.
        params_mat = mat_eng.struct(dict(self.params))
        # X_mat - (n_samples, seq_len, n_channels)
        X_mat = [matlab.double(np.ascontiguousarray(X_train, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test1, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test2, dtype=np.float64))]
        # y_mat - (n_samples, 1)
        y_mat = [matlab.double(np.ascontiguousarray(y_train, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test1, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test2, dtype=np.float64))]
        # Call `lasso_glm` matlab function.
        try:
            accuracy = mat_eng.lasso_glm(params_mat, X_mat, y_mat); accuracy = np.array(accuracy, dtype=np.float32)
            acc_test1 = np.array(accuracy[0,:], dtype=np.float32).reshape((-1,))
            acc_test2 = np.array(accuracy[1,:], dtype=np.float32).reshape((-1,))
        except matlab.engine.MatlabExecutionError as e:
            raise ValueError("ERROR: Get matlab.engine.MatlabExecutionError {}.".format(e))
        # Close the matlab engine.
        mat_eng.quit()
        # Return the final `acc_test1` & `acc_test2`.
        return acc_test1, acc_test2

# def create_toy_data func
def create_toy_data(n_samples=50, n_features=2, n_classes=5):
    """
    Create toy data from specified parameters.
    :param n_samples: The number of samples corresponding to each class.
    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :return X: (n_samples, n_features) - The source data.
    :return y: (n_samples,) - The target data.
    """
    assert n_samples >= 2 and n_features >= 2 and n_classes >= 2
    # Initialize offset according to `n_classes`.
    offset = np.arange(n_classes, dtype=np.float32)
    # Get the random data samples of each class.
    X, y = [], []
    for class_idx in range(len(offset)):
        X.append(np.random.random(size=(n_samples, n_features)).astype(np.float32) + offset[class_idx])
        y.append(np.array([class_idx for _ in range(n_samples)], dtype=np.float32))
    X = np.concatenate(X, axis=0); y = np.concatenate(y, axis=0)
    # Shuffle the original data.
    data = np.concatenate([X, y.reshape((-1,1))], axis=-1); np.random.shuffle(data)
    X = data[:,:X.shape[1]]; y = data[:,-1]
    # Return the final `X` & `y`.
    return X, y

if __name__ == "__main__":
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter
    # local dep
    from utils import DotDict
    from params.lasso_glm_params import lasso_glm_params

    # Initialize random seed.
    np.random.seed(42)
    # Initialize whether plot data.
    plot_data = False
    # Initialize image path.
    path_img = os.path.join(os.getcwd(), "__image__")
    if not os.path.exists(path_img): os.makedirs(path_img)

    # Get the [X,y] of toy data.
    X, y = create_toy_data(n_samples=50, n_features=2, n_classes=5)
    data = pd.DataFrame(DotDict({"x0":X[:,0],"x1":X[:,1],"y":y,}))
    sns.scatterplot(data=data, x="x0", y="x1", hue="y"); plt.savefig(os.path.join(path_img, "lasso_glm.data.png"))
    # Instantiate lasso_glm_params.
    lasso_glm_params_inst = lasso_glm_params(dataset="eeg")
    # Construct train-set & test1-set & test2-set using loo.
    X = np.expand_dims(X, axis=1); label_counter = Counter(y); assert (np.diff(list(label_counter.values())) == 0).all()
    test2_idxs = [np.random.choice(np.where(y == label_i)[0]) for label_i in label_counter.keys()]
    assert (len(test2_idxs) == len(set(test2_idxs))) and (len(test2_idxs) == len(label_counter.keys()))
    test1_idxs = [np.random.choice(list(
        set(np.where(y == label_i)[0]) - set(test2_idxs)
    )) for label_i in label_counter.keys()]
    assert (len(test1_idxs) == len(set(test1_idxs))) and\
        (len(test1_idxs) == len(label_counter.keys()))
    train_idxs = list(set(range(y.shape[0])) - set(test2_idxs)); assert len(train_idxs) + len(test2_idxs) == y.shape[0]
    train_idxs = list(set(range(y.shape[0])) - set(test1_idxs) - set(test2_idxs))
    assert len(train_idxs) + len(test1_idxs) + len(test2_idxs) == y.shape[0]
    # Use `train_idxs` & `test1_idxs` & `test2_idxs` to get train-set & test1-set & test2-set.
    X_train = X[train_idxs,:,:]; y_train = y[train_idxs]
    X_test1 = X[test1_idxs,:,:]; y_test1 = y[test1_idxs]
    X_test2 = X[test2_idxs,:,:]; y_test2 = y[test2_idxs]
    # Instantiate lasso_glm.
    lasso_glm_inst = lasso_glm(lasso_glm_params_inst.model)
    # Fit model with data, then log the test accuracy.
    accuracy_test1, accuracy_test2 = lasso_glm_inst.fit((X_train, X_test1, X_test2), (y_train, y_test1, y_test2))
    print((
        "The accuracy of test1-set is {:.2f}%, the accuracy of test2-set is {:.2f}%."
    ).format(np.max(accuracy_test1)*100., np.max(accuracy_test2)*100.))

