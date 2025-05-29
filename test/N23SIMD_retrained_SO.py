import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import tensorflow.keras as K
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.N23SIMD import SIMD

# ---------------------------------------------------------------------------
# Zero-shot evaluation of retrained SIMD model on sleep-EEG SO phases
#
# Workflow
#   1. Load a re-trained NREM2/3 SIMD checkpoint.
#   2. Loop through 12 subjects, eight SO phases each.
#   3. Compute per-class accuracy (random image/audio “dummy” inputs).
#   4. Persist a nested results dictionary to disk as a pickle.
# ---------------------------------------------------------------------------

def test_step(model, image_data ,audio_data, sleep_data, true_label):
    loss_value, Result_Matrix = \
          model((image_data ,audio_data, sleep_data, true_label), training = False)

    return loss_value, Result_Matrix


def load_test_data(path, batch_size, data_type):
    """
    Load pre-serialised test data for a single SO phase.

    Parameters
    ----------
    path : str
        Pickle file path containing the test set for one recording.
    batch_size : int
        Batch size used when creating the tf.data pipeline.
    data_type : str
        Key of the SO-event subclass inside the pickle file.

    Returns
    -------
    tf.data.Dataset
        Shuffled, batched, prefetched dataset of (EEG, one-hot label).
    """
    with open(path, 'rb') as file:
        dataset = pickle.load(file)

    category   = np.asarray(dataset[data_type]['label'])
    sleep_eeg  = np.asarray(dataset[data_type]['decode_data'])
    ds_length  = len(category)

    dataset_ = tf.data.Dataset.from_tensor_slices((sleep_eeg, category))
    dataset_ = dataset_.shuffle(ds_length)                       \
                       .batch(batch_size, drop_remainder=False)  \
                       .prefetch(2)
    return dataset_

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
    model_item       = SIMD(mode='normal')
    checkpoint_path  = os.path.join(checkpath, "N2N3_SIMD_Retrained.ckpt")
    model_item.load_weights(checkpoint_path)
    tf.config.run_functions_eagerly(False)   # ensure graph execution
    # -------------------------------------------------------------------------

    start_time     = time.time()
    # SO phases to evaluate
    data_type_list = [
        "Post-SO", "2nd-Up", "trans-2nd-Up", "Down",
        "trans-Down", "1st-Up", "trans-1st-Up", 'other'
    ]
    # Subjects (two-digit identifiers)
    index_numbers  = [f"{idx:02d}" for idx in range(1, 13)]

    Result_dict = {}                         # {subject: {event: accuracy}}

    # ========================= EVALUATION LOOP ===============================
    for idx in index_numbers:
        Result_dict[idx] = {}

        subj_dir   = os.path.join(recordpath, idx)
        path_test  = sorted(
            p for p in (os.path.join(subj_dir, f) for f in os.listdir(subj_dir))
            if os.path.basename(p).startswith("SO")
        )[0]

        for data_type in data_type_list:
            test_dataset   = load_test_data(path_test, batch_size, data_type)
            total_correct  = 0
            total_samples  = 0

            for sleep_eeg, labels in test_dataset:
                # SIMD expects three domains inputs; image/audio are dummies here
                dummy_img   = tf.random.normal((sleep_eeg.shape[0], 80, 55))
                dummy_audio = tf.random.normal((sleep_eeg.shape[0], 80, 55))

                _, logits   = test_step(
                    model_item, dummy_img, dummy_audio, sleep_eeg, labels
                )

                logits      = tf.reduce_mean(logits, axis=0).numpy()
                preds_eq    = np.argmax(logits, axis=-1) == np.argmax(labels, axis=-1)

                total_correct += preds_eq.sum()
                total_samples += preds_eq.size

            accuracy = total_correct / total_samples
            Result_dict[idx][data_type] = accuracy

            print(f"Subject {idx} SO-event: {data_type} Zeroshot Accuracy: {accuracy*100:.2f}%")

    # ------------------------- persist results -------------------------------
    print(f"Time taken: {time.time() - start_time:.2f}s")

    save_path = os.path.join(basepath, 'results', 'N23SIMD_retrained_SOevent_Decode_Result.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(Result_dict, f)


