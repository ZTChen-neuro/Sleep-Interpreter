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
from models.lasso_glm_temporal import lasso_glm as lasso_glm_model
import matlab.engine

"""
Awake-audio → sleep transfer-learning benchmark  
 • Trains a LASSO-regularised GLM on every single time point (80) of awake-audio EEG.  
 • Evaluates on 200-sample points from NREM2/3 and REM sleep.  
 • Heavy lifting (fitting the GLM) is done by a MATLAB function
    `lasso_glm.m` called from the Python `matlab.engine` interface.
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
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)


def load_data(path, data_type):
    """
    Load EEG data from a pkl file.

    Returns
    -------
    tuple (EEG, category)
        EEG : ndarray shape (n_samples, 200, 55)
        category : 1-D int array, class index per sample
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
    eeg        = np.asarray(eeg).transpose(0, 2, 1)

    idx = np.random.permutation(len(categories))
    categories = np.argmax(categories[idx], axis=-1)
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
    global params, paths
    # Initialize parameters & variables of current training process.
    init(params_)
    # Execute experiments for each dataset run.
    # ------------------------------ paths -----------------------------------
    basepath   = os.path.join(os.getcwd(), os.pardir, os.pardir)
    recordpath = os.path.join(basepath, 'data', 'Sleep Decoding', 'train')
    resultpath = os.path.join(basepath, 'results')
    type_list = ['awake audio']
    index_numbers   = [d for d in os.listdir(recordpath) if d.startswith('subject')]
    # ------------------------------------------------------------------------

    # ------------ start MATLAB engine ----------
    mat_eng = matlab.engine.start_matlab()
    mat_eng.addpath(os.path.join(basepath, 'code', 'model', 'lasso_glm_temporal'))
    params_mat = mat_eng.struct(dict(cp.deepcopy(params.model)))
    # -------------------------------------------
    # Prepare input for calling matlab function.
    Result_dict = {}
    for index_number in index_numbers:

        Result_dict[index_number] = {}
        NREM_check, REM_check = False, False
        for data_type in type_list:
            Result_dict[index_number][data_type] = {}
            Result_dict[index_number][data_type]['N2N3'] = []
            Result_dict[index_number][data_type]['REM'] = []


            path_record = os.path.join(recordpath, index_number)
            path_data = [os.path.join(path_record,path_i) for path_i in os.listdir(path_record)\
                            if path_i.startswith("Whole_data.pickle") ]; path_data.sort()
            path_data = path_data[0]
            
            # Run model with different train-set & test-set for `n_runs` runs, to average the accuracy curve.
            run_idx = 0; accuracies_test1 = []; accuracies_test2 = []
            while run_idx < 1:
                train_dataset = load_data(path_data, data_type)
                test_dataset1 = load_data(path_data, 'N2N3')
                test_dataset2 = load_data(path_data, 'REM')
                if len(test_dataset2) == 1:
                    REM_check = True
                    test_dataset2 = test_dataset1
                # Record the start time of preparing data.
                time_start = time.time()
                # Load data from specified experiment.
                # Get `X` & `y` from `dataset`.
                X_train, y_train = train_dataset
                X_test1, y_test1 = test_dataset1
                X_test2, y_test2 = test_dataset2
                # Train the model for each time point.

                # Check whether `X` & `y` are well-structured.
                y_train = np.expand_dims(y_train, axis=-1)
                assert (len(X_train.shape) == 3) and (len(y_train.shape) == 2)
                y_test1 = np.expand_dims(y_test1, axis=-1)
                assert (len(X_test1.shape) == 3) and (len(y_test1.shape) == 2)
                y_test2 = np.expand_dims(y_test2, axis=-1)
                assert (len(X_test2.shape) == 3) and (len(y_test2.shape) == 2)
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
                    acc_test1 = np.array(accuracy[0,:], dtype=np.float32)
                    acc_test2 = np.array(accuracy[1,:], dtype=np.float32)
                except matlab.engine.MatlabExecutionError as e:
                    raise ValueError("ERROR: Get matlab.engine.MatlabExecutionError {}.".format(e))
                accuracy_test1 = acc_test1
                accuracy_test2 = acc_test2
                # Record current time point.
                time_stop = time.time()
                # Convert `accuracy_test1` & `accuracy_test2` to `np.array`.
                accuracy_test1 = np.round(np.array(accuracy_test1, dtype=np.float32), decimals=4)
                accuracy_test2 = np.round(np.array(accuracy_test2, dtype=np.float32), decimals=4)
                accuracies_test1.append(accuracy_test1)
                accuracies_test2.append(accuracy_test2)

                # Update `run_idx` to enter next iteration.
                run_idx += 1
            
            accuracies_test1 = np.round(np.array(accuracies_test1, dtype=np.float32), decimals=4)
            accuracies_test2 = np.round(np.array(accuracies_test2, dtype=np.float32), decimals=4)
            if REM_check == True:
                Result_dict[index_number][data_type]['N2N3'] = accuracies_test1*100
                Result_dict[index_number][data_type]['REM'] = [0] * 80
            else:
                Result_dict[index_number][data_type]['N2N3'] = accuracies_test1*100
                Result_dict[index_number][data_type]['REM'] = accuracies_test2*100
            
        data_save_path = os.path.join(resultpath, "lasso_awakeaudio2sleep_temporal.pickle")
        with open(data_save_path, "wb") as f:
            pickle.dump(Result_dict, f)
    mat_eng.quit()
    

if __name__ == "__main__":
    import os
    # local dep
    from params.lasso_glm_params import lasso_glm_params

    # macro
    dataset = "eeg"
    # Initialize random seed.
    utils.model.set_seeds(1642)
    ## Instantiate lasso_glm.
    
    # Instantiate lasso_glm_params.
    lasso_glm_params_inst = lasso_glm_params(dataset=dataset)
    
    # Train lasso_glm.
    train(lasso_glm_params_inst)

