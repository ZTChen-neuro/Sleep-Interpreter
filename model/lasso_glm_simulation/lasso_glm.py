#!/usr/bin/env python3
"""
Created on 15:25, May. 15th, 2023

@author: Norbert Zheng
"""
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
            X: (10[tuple],) - The input data (X_train, X_test1, X_test2,...),
                each item is of shape (n_samples, seq_len, n_channels).
            y: (10[tuple],) - The target labels (y_train, y_test1, y_test2,...), each item is of shape (n_samples,).

        Returns:
            acc_test1: (seq_len,) - The validation accuracy of ecah asso classifier trained at each time point.
            acc_test2: (seq_len,) - The test accuracy of ecah asso classifier trained at each time point.
            ...
        """
        # Initialize `X_train` & `X_validation` & `X_test` from `X`, `y_train` & `y_validation` & `y_test` from `y`.
        X_train, X_test, X_test_2, X_test_3, X_test_4,X_test_5,X_test_6,X_test_7, X_test_8,X_test_9,X_test_10 = X 
        y_train, y_test, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7,y_test_8,y_test_9, y_test_10 = y
        # Check whether `X` & `y` are well-structured.
        y_train = np.expand_dims(y_train, axis=-1)
        assert (len(X_train.shape) == 2) and (len(y_train.shape) == 2)
        y_test = np.expand_dims(y_test, axis=-1)
        assert (len(X_test.shape) == 2) and (len(y_test.shape) == 2)
        y_test_2 = np.expand_dims(y_test_2, axis=-1)
        assert (len(X_test_2.shape) == 2) and (len(y_test_2.shape) == 2)
        y_test_3 = np.expand_dims(y_test_3, axis=-1)
        assert (len(X_test_3.shape) == 2) and (len(y_test_3.shape) == 2)
        y_test_4 = np.expand_dims(y_test_4, axis=-1)
        assert (len(X_test_4.shape) == 2) and (len(y_test_4.shape) == 2)
        y_test_5 = np.expand_dims(y_test_5, axis=-1)
        assert (len(X_test_5.shape) == 2) and (len(y_test_5.shape) == 2)
        y_test_6 = np.expand_dims(y_test_6, axis=-1)
        assert (len(X_test_6.shape) == 2) and (len(y_test_6.shape) == 2)
        y_test_7 = np.expand_dims(y_test_7, axis=-1)
        assert (len(X_test_7.shape) == 2) and (len(y_test_7.shape) == 2)
        y_test_8 = np.expand_dims(y_test_8, axis=-1)
        assert (len(X_test_8.shape) == 2) and (len(y_test_8.shape) == 2)
        y_test_9 = np.expand_dims(y_test_9, axis=-1)
        assert (len(X_test_9.shape) == 2) and (len(y_test_9.shape) == 2)
        y_test_10 = np.expand_dims(y_test_10, axis=-1)
        assert (len(X_test_10.shape) == 2) and (len(y_test_10.shape) == 2)
        
        # Start matlab engine.
        mat_eng = matlab.engine.start_matlab()
        # Add path of current directory.
        mat_eng.addpath(os.path.dirname(os.path.abspath(__file__)))
        # Prepare input for calling matlab function.
        params_mat = mat_eng.struct(dict(self.params))
        # X_mat - (n_samples, seq_len, n_channels)
        X_mat = [matlab.double(np.ascontiguousarray(X_train, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_2, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_3, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_4, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_5, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_6, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_7, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_8, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_9, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(X_test_10, dtype=np.float64))]
        # y_mat - (n_samples, 1)
        y_mat = [matlab.double(np.ascontiguousarray(y_train, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_2, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_3, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_4, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_5, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_6, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_7, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_8, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_9, dtype=np.float64)),
            matlab.double(np.ascontiguousarray(y_test_10, dtype=np.float64))]
        # Call `lasso_glm` matlab function.
        try:
            accuracy = mat_eng.lasso_glm(params_mat, X_mat, y_mat)
            accuracy = np.array(accuracy, dtype=np.float32)
            acc_test = np.array(accuracy[0], dtype=np.float32)
            acc_test_2 = np.array(accuracy[1], dtype=np.float32)
            acc_test_3 = np.array(accuracy[2], dtype=np.float32)
            acc_test_4 = np.array(accuracy[3], dtype=np.float32)
            acc_test_5 = np.array(accuracy[4], dtype=np.float32)
            acc_test_6 = np.array(accuracy[5], dtype=np.float32)
            acc_test_7 = np.array(accuracy[6], dtype=np.float32)
            acc_test_8 = np.array(accuracy[7], dtype=np.float32)
            acc_test_9 = np.array(accuracy[8], dtype=np.float32)
            acc_test_10 = np.array(accuracy[9], dtype=np.float32)
        except matlab.engine.MatlabExecutionError as e:
            raise ValueError("ERROR: Get matlab.engine.MatlabExecutionError {}.".format(e))
        # Close the matlab engine.
        mat_eng.quit()
        # Return the final `acc_validation` & `acc_test`.
        return acc_test, acc_test_2,acc_test_3,acc_test_4,acc_test_5,acc_test_6,acc_test_7,acc_test_8,acc_test_9,acc_test_10

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
    lasso_glm_params_inst = lasso_glm_params(dataset="meg_zhou2023cibr")
    # Construct train-set & validation-set & test-set using loo.
    X = np.expand_dims(X, axis=1); label_counter = Counter(y); assert (np.diff(list(label_counter.values())) == 0).all()
    test_idxs = [np.random.choice(np.where(y == label_i)[0]) for label_i in label_counter.keys()]
    assert (len(test_idxs) == len(set(test_idxs))) and (len(test_idxs) == len(label_counter.keys()))
    validation_idxs = [np.random.choice(list(
        set(np.where(y == label_i)[0]) - set(test_idxs)
    )) for label_i in label_counter.keys()]
    assert (len(validation_idxs) == len(set(validation_idxs))) and\
        (len(validation_idxs) == len(label_counter.keys()))
    train_idxs = list(set(range(y.shape[0])) - set(test_idxs)); assert len(train_idxs) + len(test_idxs) == y.shape[0]
    train_idxs = list(set(range(y.shape[0])) - set(validation_idxs) - set(test_idxs))
    assert len(train_idxs) + len(validation_idxs) + len(test_idxs) == y.shape[0]
    # Use `train_idxs` & `validation_idxs` & `test_idxs` to get train-set & validation-set & test-set.
    X_train = X[train_idxs,:,:]; y_train = y[train_idxs]
    X_validation = X[validation_idxs,:,:]; y_validation = y[validation_idxs]
    X_test = X[test_idxs,:,:]; y_test = y[test_idxs]
    # Instantiate lasso_glm.
    lasso_glm_inst = lasso_glm(lasso_glm_params_inst.model)
    # Fit model with data, then log the test accuracy.
    accuracy_validation, accuracy_test = lasso_glm_inst.fit((X_train, X_validation, X_test), (y_train, y_validation, y_test))
    print((
        "The accuracy of validation-set is {:.2f}%, the accuracy of test-set is {:.2f}%."
    ).format(np.max(accuracy_validation)*100., np.max(accuracy_test)*100.))

