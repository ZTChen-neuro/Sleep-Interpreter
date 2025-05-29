import numpy as np
import copy as cp
import pickle
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
import utils; import utils.model
from model.lasso_glm_simulation import lasso_glm as lasso_glm_model
import matlab.engine

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

def train(
    base_,                # project root
    params_,              # hyper-parameter object
    label_numbers,        # number of stimulus classes
    channels,             # EEG channels (features)
    nTrainPerStim,        # training trials per class
    nSamples,             # length of each synthetic test “recording”
    nSubj,                # synthetic “subjects” to simulate
    pattern_factor,       # variance scaling factors for the class pattern
    train_scale_list,     # variance scaling factors for training noise
    test_scale_list,      # variance scaling factors for test noise
    l1_penalty            # L1 regularisation strength
):
    """
    Simulation:

      • Generate random multivariate Gaussian noise (covariance ≠ I).  
      • Imprint class-specific patterns onto a subset of sensors.  
      • Fit a MATLAB LASSO-GLM on training data with noise scaling
        `train_scale_factor`.  
      • Evaluate on 10 independent test sets with noise scaling
        `test_scale_factor`.  
      • Repeat for every (train_scale, test_scale) combination and
        accumulate accuracy over `nSubj` synthetic subjects.
    """
    
    # Initialize parameters & variables of current training process.
    init(params_)
    # Equivalent setup


    # -----------------------------------------------------------------------
    # Launch MATLAB engine & expose the simulation folder
    # -----------------------------------------------------------------------
    mat_eng = matlab.engine.start_matlab()
    mat_eng.addpath(os.path.join(base_, 'code', 'model', 'lasso_glm_simulation'))
    params.model.l1_penalty = l1_penalty
    params_mat = mat_eng.struct(dict(cp.deepcopy(params.model)))

    Result_dict = {}

    # =========================== outer loops ===============================
    for train_scale_factor in train_scale_list:
        Result_dict[train_scale_factor] = {}
        for test_scale_factor in test_scale_list:
            Result_dict[train_scale_factor][test_scale_factor] = []
        for iSj in range(nSubj):
            print(f'iSj={iSj+1}')
            
            # generate dependence of the sensors
            A = np.random.randn(channels, channels)
            _, U = np.linalg.eig((A+A.T)/2)
            covMat = U @ np.diag(np.abs(np.random.randn(channels))) @ U.T

            # generate the true patterns
            commonPattern = pattern_factor*np.random.randn(1,channels)
            
            patterns = np.tile(commonPattern, (label_numbers,1)) + pattern_factor*np.random.randn(label_numbers,channels)

            # make training data
            trainingData = train_scale_factor*np.random.randn(label_numbers*nTrainPerStim, channels)

            a = np.tile(patterns, (nTrainPerStim, 1))

            assert (a[1] == a[1+label_numbers]).all()
            # Add patterns to training data (excluding null examples)
            trainingData += np.tile(patterns, (nTrainPerStim, 1))
            trainingLabels = np.tile(np.arange(0,label_numbers), nTrainPerStim)


            # # Add more noise to some patterns
            MoreNoiseind = np.random.choice(np.arange(1,label_numbers+1), 10, replace=False)
            indend = MoreNoiseind*nTrainPerStim
            indstart = (MoreNoiseind-1)*nTrainPerStim
            
            nindex = []
            for iind in range(len(MoreNoiseind)):
                xtemp = np.arange(indstart[iind], indend[iind]) # -1 for zero-based
                nindex.extend(xtemp)
            nindex = np.array(nindex, dtype=int)
            
            trainingData[nindex, :] += train_scale_factor*np.random.randn(len(nindex), channels)
            
            # make long unlabelled data
            Test_Dataset = []
            Test_label = []
            for test_scale_factor in test_scale_list:
                Test = np.zeros((nSamples, channels))
                Test[0,:] = test_scale_factor*np.random.randn(channels)
                for iT in range(1,nSamples):
                    Test[iT,:] = 0.95*(Test[iT-1,:] + np.random.multivariate_normal(np.zeros(channels), covMat))

                testlabel = []
                for point in range(nSamples):
                    point_label = np.random.randint(0,label_numbers)
                    Test[point,:] += patterns[point_label,:]
                    testlabel.append(point_label)
                Test_Dataset.append(Test)
                Test_label.append(testlabel)
                
            
            X_train, X_test, X_test_2, X_test_3, X_test_4,X_test_5,X_test_6,X_test_7, X_test_8,X_test_9,X_test_10 = trainingData, Test_Dataset[0], Test_Dataset[1],Test_Dataset[2],Test_Dataset[3],Test_Dataset[4],\
                        Test_Dataset[5], Test_Dataset[6],Test_Dataset[7],Test_Dataset[8],Test_Dataset[9]
            y_train, y_test, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7,y_test_8,y_test_9, y_test_10 = trainingLabels, Test_label[0],\
                            Test_label[1],Test_label[2],Test_label[3],Test_label[4],\
                        Test_label[5], Test_label[6],Test_label[7],Test_label[8],Test_label[9]
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
            
            # Return the final `acc_validation` & `acc_test`.
            acc_1, acc_2,acc_3,acc_4,acc_5,acc_6,acc_7,acc_8,acc_9,acc_10 =\
                acc_test, acc_test_2,acc_test_3,acc_test_4,acc_test_5,acc_test_6,acc_test_7,acc_test_8,acc_test_9,acc_test_10
            # Convert `accuracy_validation` & `accuracy_test` to `np.array`.
            assert len(acc_1) == 1
            acc_1 = np.round(np.array(acc_1, dtype=np.float32)[0], decimals=4)
            acc_2 = np.round(np.array(acc_2, dtype=np.float32)[0], decimals=4)
            acc_3 = np.round(np.array(acc_3, dtype=np.float32)[0], decimals=4)
            acc_4 = np.round(np.array(acc_4, dtype=np.float32)[0], decimals=4)
            acc_5 = np.round(np.array(acc_5, dtype=np.float32)[0], decimals=4)
            acc_6 = np.round(np.array(acc_6, dtype=np.float32)[0], decimals=4)
            acc_7 = np.round(np.array(acc_7, dtype=np.float32)[0], decimals=4)
            acc_8 = np.round(np.array(acc_8, dtype=np.float32)[0], decimals=4)
            acc_9 = np.round(np.array(acc_9, dtype=np.float32)[0], decimals=4)
            acc_10 = np.round(np.array(acc_10, dtype=np.float32)[0], decimals=4)
            Accuracy = [acc_1, acc_2,acc_3,acc_4,acc_5,acc_6,acc_7,acc_8,acc_9,acc_10]
            for _,test_scale_factor in enumerate(test_scale_list):
                Result_dict[train_scale_factor][test_scale_factor].append(Accuracy[_]*100)
        data_save_path = os.path.join(base_, "results", "Simulation_result","Simulation_"+str(pattern_factor)+'_'+str(l1_penalty)+".pickle")
        with open(data_save_path, 'wb') as f:
            pickle.dump(Result_dict, f)
    # Close the matlab engine.
    mat_eng.quit()


    
    
if __name__ == "__main__":
    import os
    from params.lasso_glm_params import lasso_glm_params

    # Initialize random seed.
    utils.model.set_seeds(503)  # reproducibility
    
    # ---------------- simulation hyper-parameters ---------------------------
    label_numbers   = 15
    channels        = 55
    nTrainPerStim   = 60
    nSamples        = 1500
    nSubj           = 135
    train_scales    = [1, 2, 4, 5, 10, 20, 40, 50, 100, 200]
    test_scales     = train_scales                        # symmetric grid
    base        = os.path.join(os.getcwd(), os.pardir, os.pardir)

    l1_list = [               # sweep a wide range of λ values
        *np.arange(1e-5, 1e-4, 1e-5),
        *np.arange(1e-4, 1e-3, 1e-4),
        *np.arange(1e-3, 1e-2, 1e-3),
        *np.arange(1e-2, 1e-1, 1e-2)
    ]
    pattern_factors = [0.5, 1, 2, 2.5, 4, 5, 10]
    for l1_penalty in l1_list:
        for pf in pattern_factors:
            print(f"λ={l1_penalty:.0e}, pattern_factor={pf}")
            lasso_params = lasso_glm_params()
            train(
                base, lasso_params, label_numbers, channels,
                nTrainPerStim, nSamples, nSubj,
                pf, train_scales, test_scales, l1_penalty
            )