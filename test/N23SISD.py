import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import tensorflow.keras as K
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.N23SISD import SISD

# ---------------------------------------------------------------------------
# Zero-shot evaluation of SISD model on NREM2/3 sleep-EEG data
#
# Workflow
#   1. Load a pretrained SISD model checkpoint.
#   2. Loop through 12 subjects.
#   3. Compute per-class accuracy.
#   4. Persist a nested results dictionary to disk as a pickle.
# ---------------------------------------------------------------------------


def test_step(model, sleep_data, true_label):
    loss_value, Result_Matrix = \
          model((sleep_data, true_label), training = False)

    return loss_value, Result_Matrix


def load_test_data(path, batch_size):
    """
    Load pre-serialised test data for a subject's NREM2/3 data.

    Parameters
    ----------
    path : str
        Pickle file path containing the test set for one recording.
    batch_size : int
        Batch size used when creating the tf.data pipeline.

    Returns
    -------
    tf.data.Dataset
        Shuffled, batched, prefetched dataset of (EEG, one-hot label).
    """
    with open(path, 'rb') as file:
        dataset = pickle.load(file)

    categories, sleep_eeg = [], []
    for entry in dataset:
        categories.append(entry['category'])
        sleep_eeg.append(entry['Sleep_EEG'])

    categories = np.asarray(categories)
    sleep_eeg  = np.asarray(sleep_eeg)
    sleep_eeg  = np.transpose(sleep_eeg, (0, 2, 1))  # (batch, time, channels)

    ds_length = len(categories)
    Dataset = tf.data.Dataset.from_tensor_slices((sleep_eeg, categories))
    Dataset = Dataset.shuffle(ds_length)                      \
           .batch(batch_size, drop_remainder=False) \
           .prefetch(2)
    return Dataset

if __name__ == "__main__":
    # ----------------------- configurable hyper-params ------------------------
    batch_size            = 2048
    shuffle_buffer_size   = batch_size
    cycle_length          = 1           
    utils.model.set_seeds(1642)         
    # -------------------------------------------------------------------------

    # --------------------------- filesystem paths ----------------------------
    basepath     = os.path.join(os.getcwd(), os.pardir, os.pardir)
    checkpath    = os.path.join(basepath, 'data', 'Model checkpoint', 'Decoding')
    recordpath   = os.path.join(basepath, 'data', 'Sleep Decoding', 'test')
    # -------------------------------------------------------------------------

    # ------------------------- model initialisation --------------------------
    model_item       = SISD()
    checkpoint_path  = os.path.join(checkpath, "N2N3_SISD.ckpt")
    model_item.load_weights(checkpoint_path)
    tf.config.run_functions_eagerly(False)   # ensure graph execution
    # -------------------------------------------------------------------------

    start_time   = time.time()
    # Subjects (two-digit identifiers)
    subject_ids  = [f"{i:02d}" for i in range(1, 13)]
    results_dict = {}   # {subject_id: accuracy}
    accuracies   = []   # list for averaging

    # ========================= EVALUATION LOOP ===============================
    for sid in subject_ids:
        # Locate the sole REM pickle for this subject
        subj_dir  = os.path.join(recordpath, sid)
        data_file  = sorted(
            p for p in (os.path.join(subj_dir, f) for f in os.listdir(subj_dir))
            if os.path.basename(p).startswith("N2N3")
        )[0]

        dataset        = load_test_data(data_file, batch_size)
        correct_total  = 0
        sample_total   = 0
        total_loss     = 0.0

        # ---------- batch loop ----------
        for sleep_eeg, labels in dataset:

            loss, logits = test_step(
                model_item, sleep_eeg, labels
            )

            logits     = tf.reduce_mean(logits, axis=0).numpy()
            preds_eq   = np.argmax(logits, axis=-1) == np.argmax(labels, axis=-1)
            correct_total += preds_eq.sum()
            sample_total  += preds_eq.size
            total_loss    += loss
        # ---------------------------------

        accuracy = correct_total / sample_total
        results_dict[sid] = accuracy
        accuracies.append(accuracy)
        print(f"Subject {sid} Zeroshot Accuracy: {accuracy*100:.2f}%")

    # ------------------------ aggregate statistics --------------------------
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"Average Zeroshot Accuracy: {mean_acc*100:.2f}%")
    print(f"Time taken: {time.time() - start_time:.2f}s")

    results_dict['Average'] = mean_acc
    # ----------------------------- save -------------------------------------
    save_path = os.path.join(basepath, 'results', 'N23SISD_Decode_Result.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
