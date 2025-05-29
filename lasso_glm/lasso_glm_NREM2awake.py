#!/usr/bin/env python3
import time
import copy as cp
import numpy as np
import pickle
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
import utils; import utils.model
from model.lasso_glm import lasso_glm as lasso_glm_model

"""
NREM-to-Awake transfer-learning benchmark using a LASSO-regularised GLM.

Pipeline
--------
1.  Load awake EEG (image or audio paradigm) as training data.  
2.  For each subject and every 2-s data sequence in the NREM data for training,
    slice an 800-ms window to calculate test sets.  
3.  Fit a LASSO-GLM (one run per offset) and record test accuracies.  
4.  Store all accuracies in a nested `Result_dict` → pickle.
"""

"""
init funcs
"""
# def init func
def init(params_):
    """
    Initialize `lasso_glm` training variables.

    Args:
        params_: The parameters of current training process.

    Returns:
        None
    """
    global params
    # Initialize params.
    params = cp.deepcopy(params_)
    

def load_train_data(path, data_type, time_point):
    """
    Extract an *8-s window* from the N2N3 sleep stage.

    Parameters
    ----------
    time_point : int
        Start index in samples (0…200) → slice [t : t+80].

    Returns
    -------
    tuple (eeg, category) or (0,)
        If no data exist for the requested phase, returns (0,)
    """
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    categories, eeg = [], []
    for item in dataset:
        categories.append(item['category'])
        eeg.append(item[data_type])

    if len(eeg)==0:             
        return (0,)

    categories = np.asarray(categories)
    eeg        = np.asarray(eeg).transpose(0, 2, 1)       # (batch, 200, 55)
    eeg        = eeg[:, time_point:time_point+80, :]      # (batch, 80, 55)
    assert eeg.shape[1:] == (80, 55)

    idx = np.random.permutation(len(categories))
    categories = np.argmax(categories[idx], axis=-1)
    eeg        = eeg[idx]

    return (eeg, categories)


def load_test_data(path, data_type):
    """
    Load *awake* EEG data from a pkl file.

    Returns
    -------
    tuple (EEG, category)
        EEG : ndarray shape (n_samples, 80, 55)
        category : 1-D int array, class index per sample
    """
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    categories, eeg = [], []
    for item in dataset:
        categories.append(item['category'])
        eeg.append(item[data_type])

    categories = np.asarray(categories)
    eeg        = np.asarray(eeg).transpose(0, 2, 1)      # (batch, 80, 55)

    # Randomise sample order
    idx = np.random.permutation(len(categories))
    categories = np.argmax(categories[idx], axis=-1)      # one-hot → int
    eeg        = eeg[idx]

    return (eeg, categories)

    

"""
train funcs
"""
# def train func
def train(params_):
    """
    Train the model.

    Args:
        params_: The parameters of current training process.

    Returns:
        None
    """
    global params
    # Initialize parameters & variables of current training process.
    init(params_)
    # Execute experiments for each dataset run.
        # ------------------------------ paths -----------------------------------
    basepath   = os.path.join(os.getcwd(), os.pardir, os.pardir)
    recordpath = os.path.join(basepath, 'data', 'Sleep Decoding', 'train')
    resultpath = os.path.join(basepath, 'results')
    type_list  = ['awake image', 'awake audio']        # two awake paradigms
    index_numbers   = [d for d in os.listdir(recordpath) if d.startswith('subject')]
    start_time_point_list = list(range(0, 130, 10))          # 0-120 s in 10-s steps
    # ------------------------------------------------------------------------
    Result_dict = {}
    for index_number in index_numbers:
        Result_dict[index_number] = {}
        for time_point in start_time_point_list:
            Result_dict[index_number][time_point] = {}
            Result_dict[index_number][time_point][type_list[0]] = []
            Result_dict[index_number][time_point][type_list[1]] = []

            path_record = os.path.join(recordpath, index_number)
            path_data = [os.path.join(path_record,path_i) for path_i in os.listdir(path_record)\
                            if path_i.startswith("Whole_data.pickle") ]; path_data.sort()
            path_data = path_data[0]
            train_dataset = load_train_data(path_data, 'N2N3', time_point)
            if len(train_dataset) == 1:
                Result_dict[index_number][time_point][type_list[0]] = [0] * 80
                Result_dict[index_number][time_point][type_list[1]] = [0] * 80
                continue
            

            test_dataset1 = load_test_data(path_data, type_list[0])
            test_dataset2 = load_test_data(path_data, type_list[1])
            # Run model with different train-set & test-set for `n_runs` runs, to average the accuracy curve.
            run_idx = 0; accuracies_test1 = []; accuracies_test2 = []
            while run_idx < 1:
                # Record the start time of preparing data.
                time_start = time.time()
                # Get `X` & `y` from `dataset`.
                X_train, y_train = train_dataset
                X_test1, y_test1 = test_dataset1
                X_test2, y_test2 = test_dataset2
                # Train the model for each time point.
                model = lasso_glm_model(params.model)
                try:
                    accuracy_test1, accuracy_test2 =\
                        model.fit((X_train, X_test1, X_test2), (y_train, y_test1, y_test2))
                except ValueError as e:
                    msg = (
                        "ERROR: Get ValueError {}, re-run current run."
                    ).format(e); print(msg); continue
                # Append `accuracy` to `accuracies`.
                accuracies_test1.append(accuracy_test1); accuracies_test2.append(accuracy_test2)
                # Record current time point.
                time_stop = time.time()
                # Convert `accuracy_test1` & `accuracy_test2` to `np.array`.
                accuracy_test1 = np.round(np.array(accuracy_test1, dtype=np.float32), decimals=4)
                accuracy_test2 = np.round(np.array(accuracy_test2, dtype=np.float32), decimals=4)
                
                for time_idx, (accuracy_test1_i, accuracy_test2_i) in enumerate(zip(accuracy_test1, accuracy_test2)):
                    msg += (
                        "\nGet test-accuracy1 ({:.2f}%) and test-accuracy2 ({:.2f}%) at time index {:d}."
                    ).format(accuracy_test1_i*100., accuracy_test2_i*100., time_idx)
                print(msg)
                # Update `run_idx` to enter next iteration.
                run_idx += 1
            # Average `accuracies` to get `avg_accuracy`.
            avg_accuracy_test1 = np.mean(accuracies_test1, axis=0)
            avg_accuracy_test2 = np.mean(accuracies_test2, axis=0)
            # Convert `avg_accuracy_test1` & `avg_accuracy_test2` to `np.array`.
            avg_accuracy_test1 = np.round(np.array(avg_accuracy_test1, dtype=np.float32), decimals=4)
            avg_accuracy_test2 = np.round(np.array(avg_accuracy_test2, dtype=np.float32), decimals=4)
            Result_dict[index_number][time_point][type_list[0]].append(avg_accuracy_test1*100)
            Result_dict[index_number][time_point][type_list[1]].append(avg_accuracy_test2*100)
        data_save_path = os.path.join(resultpath, "lasso_NREM2awake.pickle")
        save_file = open(data_save_path,'wb')
        pickle.dump(Result_dict,save_file)
        save_file.close()
    

if __name__ == "__main__":
    import os
    from params.lasso_glm_params import lasso_glm_params

    # macro
    dataset = "eeg"
    # Initialize random seed.
    utils.model.set_seeds(1642)

    ## Instantiate lasso_glm.
    # Train lasso_glm.
    lasso_glm_params_inst = lasso_glm_params(dataset=dataset)
    # Train lasso_glm.
    train(lasso_glm_params_inst)

