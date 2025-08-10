import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import pandas as pd
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    from model.SIStaging.Staging import Staging

# ---------------------------------------------------------------------------
# SI automatic sleep-staging evaluation script
#
#   • Loads a pre-trained Staging model (500 Hz input, subject-independent).
#   • Iterates over every “subjectXX/whole*.pkl” recording in the test set.
#   • Computes epoch-level accuracy, then upsamples to 0.1-epoch resolution
#     (×10) so it can be directly compared to the provided raw labels.
#   • Saves the per-subject prediction matrix + ground truth to disk.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
basepath   = os.path.join(os.getcwd(), os.pardir)
datapath   = os.path.join(basepath, os.pardir, 'data', 'SI Staging')
checkpath  = os.path.join(basepath, os.pardir, 'data', 'Model checkpoint', 'Staging')

# ---------------------------------------------------------------------------
# Instantiate model and load checkpoint
# ---------------------------------------------------------------------------
weights = [                                     # class-balancing loss weights
    [1.1, 1.2, 1.0, 1.0],  # fold-1
    [1.1, 1.0, 1.2, 1.0],  # fold-2
    [1.1, 1.0, 1.0, 1.1],  # fold-3
    [1.0, 1.2, 1.2, 1.0],  # fold-4
    [1.0, 1.2, 1.0, 1.1],  # fold-5
    [1.0, 1.0, 1.2, 1.1],  # fold-6
]
Model = Staging(fs=500, weights=weights)
checkpoint_path = os.path.join(checkpath, "Staging_model.ckpt")
Model.load_weights(checkpoint_path).expect_partial()

# ---------------------------------------------------------------------------
# Build test-set file lists
# ---------------------------------------------------------------------------
path_test_file = os.path.join(datapath, "Test")
path_test      = sorted(
    os.path.join(path_test_file, p) for p in os.listdir(path_test_file)
    if p.startswith("subject")
)
# Per-subject: “whole*.pkl” (EEG) and “raw*.pkl” (pre-segmented labels)
path_test_data  = sorted(
    os.path.join(path_test, p) for p in os.listdir(path_test)
    if p.startswith("Whole")
)
path_test_label = sorted(
    os.path.join(path_test, p) for p in os.listdir(path_test)
    if p.startswith("raw")
)


# Map labels to one-hot indices (Wake, N1, N2, N3, REM → 0-3 after merge)
index_dict = {"W": 0, "N1": 0, "N2": 1, "N3": 2, "R": 3}

Accuracy     = []   # per-subject correct sample counts
Result_dict  = {}   # per-subject detailed results
count        = 0    # total number of subjects processed

# ---------------------------------------------------------------------------
# ---------------------------  main evaluation ------------------------------
# ---------------------------------------------------------------------------
for i in range(len(path_test_data)):

    # ---------- load data ----------
    with open(path_test_data[i], "rb") as f:
        dataset = pickle.load(f)
    with open(path_test_label[i], "rb") as ff:
        label_data = pickle.load(ff)

    # ---------- unpack pickle content ----------
    category, label, sleep_data = [], [], []
    for _ in range(len(dataset)):
        label.append(dataset[_]['label'])                # integer class
        category.append(np.eye(4)[dataset[_]['label']])  # one-hot vector
        sleep_data.append(dataset[_]['data'])            # EEG segment

    category   = np.asarray(category)
    label      = np.asarray(label)
    sleep_data = np.asarray(sleep_data)

    # ---------- forward pass ----------
    loss_value, Result_Matrix1 = Model((sleep_data, category, label),
                                       training=False)

    # ---------- upsample raw hypnogram (30 s → 3 s resolution) ----------
    raw_label   = label_data
    New_label   = []
    for i in range(len(label_data)):
        New_label += [label_data[i]] * 10                # repeat each epoch 10×
    assert len(label_data) * 10 - 9 == len(label)        # sanity check

    # ---------- post-process network output ----------
    Result_Matrix1 = dataset['result matrix']            # stored logits
    for i in range(1, len(Result_Matrix1)):              # temporal smoothing
        Result_Matrix1[i] = Result_Matrix1[i-1] * 0.9 + 0.1 * Result_Matrix1[i]

    pred_label  = np.argmax(Result_Matrix1, axis=-1)     # predicted class
    pred_add    = pred_label[-1].repeat(9)               # pad tail to 3 s grid
    pred_label  = np.append(pred_label, pred_add)
    assert len(pred_label) == len(New_label)

    # ---------- compute accuracy ----------
    New_label    = np.asarray(New_label)
    accuracy_vec = pred_label == New_label
    accuracy     = np.sum(accuracy_vec) / accuracy_vec.size

    print(f'Subject {i} Staging Accuracy: {accuracy*100:.2f}%')

    # ---------- collect results ----------
    result_dict = {
        'RT': pred_label,          # runtime prediction (upsampled)
        'GT': New_label,           # ground-truth label (upsampled)
        'result matrix': Result_Matrix1
    }
    Result_dict[i] = result_dict
    Accuracy.append(np.sum(accuracy_vec))
    count += 1

# ---------------------------------------------------------------------------
# Summary across all subjects
# ---------------------------------------------------------------------------
Whole_acc = sum(Accuracy) / count * 100
print(f'Whole Subjects Averaged Staging Accuracy: {Whole_acc:.2f}%')

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
result_save_path = os.path.join(basepath, 'results', 'Staging results')
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)

result_save_file = open(os.path.join(result_save_path, 'Result_Matrix.pickle'), 'wb')
pickle.dump(Result_dict, result_save_file)
result_save_file.close()
