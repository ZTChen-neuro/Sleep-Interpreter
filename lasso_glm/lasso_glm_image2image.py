#!/usr/bin/env python3

import time
import copy as cp
import numpy as np
import pickle

# local dep
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.pardir)

import utils
import utils.model
import matlab.engine


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _build_5fold_partitions(n_items):
    """
    Partition a range of indices into 5 folds as evenly as possible.

    This returns a list of 5 numpy arrays with (approximately) equal sizes.

    Args:
        n_items (int): Dataset length.

    Returns:
        List[np.ndarray]: Five arrays of indices covering 0..n_items-1.
    """
    fold_sizes = [n_items // 5] * 5
    remainder = n_items % 5
    # Distribute the remainder across the first folds for balance
    for i in range(remainder):
        fold_sizes[i] += 1
    indices = np.arange(n_items)
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size
    return folds


def load_data(path, data_type):
    """Load data and produce a 5-fold (train/val/test) split.

    Strategy:
        - Shuffle all items.
        - Partition into 5 folds.
        - Use fold 0 as **test**, fold 1 as **validation**, remaining three
          folds concatenated as **train**.

    Args:
        path (str): Path to the pickle dataset file.
        data_type (str): One of {"awake_image", "awake_audio", "N2N3", "REM"}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (EEG, labels) where EEG has
        shape (N, T, C) and labels is a 1D integer array of shape (N,).
    """
    with open(path, "rb") as file:
        Dataset = pickle.load(file)
    dataset = Dataset[data_type]
    category, EEG = [], []

    # Extract targets and features
    for _ in range(len(dataset)):
        category.append(dataset[_]["category"])
        EEG.append(dataset[_]["EEG"])
    if len(EEG) ==0:
        return [0],[0]
    else:
        category = np.asarray(category)
        EEG = np.asarray(EEG)

        # Expect raw shape (B, C, T) -> transpose to (B, T, C)
        EEG = np.transpose(EEG, (0, 2, 1))

        # Shuffle indices and construct 5 folds
        dataset_length = len(category)
        dataset_index = np.arange(dataset_length)
        np.random.shuffle(dataset_index)

        category = category[dataset_index]
        EEG = EEG[dataset_index]

        # Convert one-hot to integer class indices
        category = np.argmax(category, axis=-1)

        return EEG, category


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(base_, params_):
    """
    Train the model (driver function).

    Args:
        base_ (str): Base path of the current project.
        params_ : Parameters object for the current training process.

    Returns:
        None
    """
    global params
    params = cp.deepcopy(params_)
    # Resolve base paths
    basepath = os.path.join(os.getcwd(), os.pardir)
    recordpath = os.path.join(base_, os.pardir, "data", 'SI Main Train')
    recordpath2 = os.path.join(base_, os.pardir, "data", 'SI Main Test')
    resultpath = os.path.join(basepath, 'results')

    # Data type configuration
    type_list = ["awake_image", "awake_audio", "N2N3", "REM"]
    data_type = type_list[0]

    # Start MATLAB engine and add the model path
    mat_eng = matlab.engine.start_matlab()
    mat_eng.addpath(
        os.path.abspath(os.path.join(basepath, "model", "lasso_glm"))
    )

    # Prepare MATLAB struct from params for the lasso_glm call
    params_mat = mat_eng.struct(dict(cp.deepcopy(params.model)))

    # ------------------------------------------------------------------------
    Result_dict = {}
    index_numbers   = [d for d in os.listdir(recordpath) if d.startswith('subject')]
    index_numbers   += [d for d in os.listdir(recordpath2) if d.startswith('subject')]
    index_numbers.sort()
    path_records = [os.path.join(recordpath, path_i) for path_i in os.listdir(recordpath) if path_i.startswith('subject')]
    path_records += [os.path.join(recordpath2, path_i) for path_i in os.listdir(recordpath2) if path_i.startswith('subject')]
    path_records.sort()
    for index, index_number in enumerate(index_numbers):

        Result_dict[index_number] = []

        path_record = path_records[index]
        path_data = [os.path.join(path_record,path_i) for path_i in os.listdir(path_record)\
                        if path_i.startswith("Whole_data.pickle") ]; path_data.sort()
        path_data = path_data[0]

        X_all, y_all = load_data(path_data, data_type)

        # No EEG data
        if len(X_train)==1 and len(y_train) == 1:
            Result_dict[index_number] = [[0,0]] * 5
            continue

        n_items = X_all.shape[0]

        # 5-fold split
        folds = _build_5fold_partitions(n_items)

        run_idx = 0
        accuracies_validation = []
        accuracies_test = []

        while run_idx < 5:
            # Start time for profiling
            time_start = time.time()

            test_idx = folds[run_idx]
            val_idx = folds[(run_idx + 1) % 5]
            train_idx = np.concatenate([folds[j] for j in range(5) if j not in {run_idx, (run_idx + 1) % 5}], axis=0)

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_validation, y_validation = X_all[val_idx], y_all[val_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

            # Expand labels to (N, 1) for MATLAB
            y_train = np.expand_dims(y_train, axis=-1)
            assert (len(X_train.shape) == 3) and (len(y_train.shape) == 2)

            y_validation = np.expand_dims(y_validation, axis=-1)
            assert (len(X_validation.shape) == 3) and (len(y_validation.shape) == 2)

            y_test = np.expand_dims(y_test, axis=-1)
            assert (len(X_test.shape) == 3) and (len(y_test.shape) == 2)

            # Prepare MATLAB inputs: (n_samples, seq_len, n_channels)
            X_mat = [
                matlab.double(np.ascontiguousarray(X_train, dtype=np.float64)),
                matlab.double(np.ascontiguousarray(X_validation, dtype=np.float64)),
                matlab.double(np.ascontiguousarray(X_test, dtype=np.float64)),
            ]
            # Labels: (n_samples, 1)
            y_mat = [
                matlab.double(np.ascontiguousarray(y_train, dtype=np.float64)),
                matlab.double(np.ascontiguousarray(y_validation, dtype=np.float64)),
                matlab.double(np.ascontiguousarray(y_test, dtype=np.float64)),
            ]

            # Call MATLAB lasso_glm
            try:
                accuracy = mat_eng.lasso_glm(params_mat, X_mat, y_mat)
                accuracy = np.array(accuracy, dtype=np.float32)
                acc_validation = np.array(accuracy[0, :], dtype=np.float32).reshape(
                    (-1,)
                )
                acc_test = np.array(accuracy[1, :], dtype=np.float32).reshape((-1,))
            except matlab.engine.MatlabExecutionError as e:
                raise ValueError(
                    f"ERROR: Get matlab.engine.MatlabExecutionError {e}."
                )

            accuracy_validation = acc_validation
            accuracy_test = acc_test

            # Log for aggregation
            accuracies_validation.append(accuracy_validation)
            accuracies_test.append(accuracy_test)

            time_stop = time.time()

            # Round for printing and result storage
            accuracy_validation = np.round(
                np.array(accuracy_validation, dtype=np.float32), decimals=4
            )
            accuracy_test = np.round(
                np.array(accuracy_test, dtype=np.float32), decimals=4
            )

            Result_dict[index_number].append([accuracy_validation * 100, accuracy_test * 100])

            time_maxacc_idxs = np.where(
                accuracy_validation == np.max(accuracy_validation)
            )[0]
            time_maxacc_idx = time_maxacc_idxs[
                np.argmax(accuracy_test[time_maxacc_idxs])
            ]

            # Verbose run log (kept)
            msg = (
                "Finish run index {:d} in {:.2f} seconds, with test-accuracy (" "{:.2f}%)"
                + " according to max validation-accuracy ({:.2f}%) at time index {:d}."
            ).format(
                run_idx,
                time_stop - time_start,
                accuracy_test[time_maxacc_idx] * 100.0,
                accuracy_validation[time_maxacc_idx] * 100.0,
                time_maxacc_idx,
            )
            print(msg)

            run_idx += 1

        # Average across runs
        avg_accuracy_validation = np.mean(accuracies_validation, axis=0)
        avg_accuracy_test = np.mean(accuracies_test, axis=0)

        # Round for saving
        avg_accuracy_validation = np.round(
            np.array(avg_accuracy_validation, dtype=np.float32), decimals=4
        )
        avg_accuracy_test = np.round(
            np.array(avg_accuracy_test, dtype=np.float32), decimals=4
        )

        data_save_path = os.path.join(
            resultpath, "lasso_image2image.pickle"
        )
        with open(data_save_path, "wb") as f:
            pickle.dump(Result_dict, f)

    mat_eng.quit()


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # local dep
    from params.lasso_glm_params import lasso_glm_params

    # macro
    dataset = "eeg"

    # Initialize random seed.
    utils.model.set_seeds(1642)

    # Instantiate lasso_glm.
    base = os.path.join(os.getcwd(), os.pardir)
    lasso_glm_params_inst = lasso_glm_params(dataset=dataset)

    # Train lasso_glm.
    train(base, lasso_glm_params_inst)
