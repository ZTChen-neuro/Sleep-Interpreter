import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import copy as cp
from collections import Counter
# from tensorflow import keras
import tensorflow.keras as K
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.REMSIMD import SIMD

# ---------------------------------------------------------------------------
# Multi-domain REM Sleep Decoder – Fine-tuning Script
# ---------------------------------------------------------------------------
# This script fine-tunes a *multi-encoder* model that combines image, audio
# and EEG inputs to classify REM-stage sleep data.  
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------#
# One training / validation / test step 
# ---------------------------------------------------------------------------#

def train_step(model_decode, image_data ,audio_data, sleep_data, true_label, optimizer):
    """Forward + backward pass on one mini-batch."""
    with tf.GradientTape() as tape:
        loss_value, Result_Matrix \
            = model_decode((image_data ,audio_data, sleep_data, true_label), training=True)

    grads = tape.gradient(loss_value, model_decode.trainable_variables)

    optimizer.apply_gradients(zip(grads, model_decode.trainable_variables))
    return loss_value, Result_Matrix

def validation_step(model_decode, image_data ,audio_data, sleep_data, true_label):

    """Validation forward pass (no gradient tracking)."""
    loss_value, Result_Matrix =\
          model_decode((image_data ,audio_data, sleep_data, true_label), training=False)

    return loss_value, Result_Matrix


def test_step(model, image_data ,audio_data, sleep_data, true_label):
    """Validation forward pass (no gradient tracking)."""
    loss_value, Result_Matrix = \
          model((image_data ,audio_data, sleep_data, true_label), training = False)

    return loss_value, Result_Matrix

# ---------------------------------------------------------------------------#
# Dataset loader: pickle → tf.data.Dataset
# ---------------------------------------------------------------------------#
def load_data(path, batch_size, data_type):
    """
    Load samples of a specified *data_type* ('train'/'val'/'test') from a
    pickle file and return a shuffled, batched `tf.data.Dataset`.

    The function performs:
      1. Field extraction
      2. Axis re-ordering to match model expectations
      3. Random temporal slice selection for image & audio streams
    """
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    image_data, audio_data, category, sleep_eeg = [], [], [], []
    for sample in dataset:
        if sample['data_type'] != data_type:
            continue
        category.append(sample['category'])
        sleep_eeg.append(sample['Sleep_EEG'])
        image_data.append(sample['awake_image_erp'])
        audio_data.append(sample['awake_audio_erp'])

    # Convert to numpy for efficient vectorised ops
    category   = np.asarray(category)
    sleep_eeg  = np.asarray(sleep_eeg)
    awake_img  = np.asarray(image_data)
    awake_aud  = np.asarray(audio_data)

    # Re-order axes: EEG → (batch, time, channel)
    sleep_eeg  = np.transpose(sleep_eeg, (0, 2, 1))
    awake_img  = np.transpose(awake_img, (0, 1, 3, 2))
    awake_aud  = np.transpose(awake_aud, (0, 1, 3, 2))

    # Randomly sample **one** 
    new_index = np.random.choice(np.arange(15), 1, replace=False)
    awake_img = tf.gather(awake_img, new_index, axis=1)
    awake_aud = tf.gather(awake_aud, new_index, axis=1)

    dataset_tf = tf.data.Dataset.from_tensor_slices(
        (awake_img, awake_aud, sleep_eeg, category)
    )
    dataset_tf = dataset_tf.shuffle(len(category)).batch(
        batch_size, drop_remainder=False
    ).prefetch(2)
    return dataset_tf

# ---------------------------------------------------------------------------#
# Main fine-tuning routine
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    # -------------------- Reproducibility & paths -------------------- #
    utils.model.set_seeds(1642)
    basepath  = os.path.join(os.getcwd(), os.pardir)
    recordpath = os.path.join(basepath, 'data', 'SI Main Test')
    checkpath  = os.path.join(basepath, 'data', 'Model checkpoint', 'Decoding')

    # -------------------- Hyper-parameters --------------------------- #
    batch_size  = 64
    finetune_type = "awake sleep finetune"                  
    epochs = 200                                

    # List of test subjects
    index_numbers = ["subject_test_"+str(i) for i in range(12)]

    # -------------------- Results dict ------------------------------- #
    Show_result = {}

    # ----------------------------------------------------------------- #
    # Leave-one-subject-out fine-tuning loop
    # ----------------------------------------------------------------- #
    for subject_id in index_numbers:
        # ----- Load pre-trained SIMD model weights ----- #
        model = SIMD(finetune=True)
        ckpt = os.path.join(checkpath,
                            f"REM_SIMD_model.ckpt")
        model.load_weights(ckpt)

        # ----- Prepare tf.data datasets for this subject ----- #
        subject_dir = os.path.join(recordpath, subject_id)
        # The REM pickle filename is the only file starting with "REM"
        path_data = sorted(
            f for f in (os.path.join(subject_dir, p)
                        for p in os.listdir(subject_dir))
            if os.path.basename(f).startswith("REM")
        )[0]

        train_ds = load_data(path_data, batch_size, 'train')
        val_ds   = load_data(path_data, batch_size, 'val')
        test_ds  = load_data(path_data, batch_size, 'test')

        # ----- Optimizer ----- #
        optimizer = K.optimizers.Adam(learning_rate=2e-5)

        # ----- History containers ----- #
        train_loss_hist, val_loss_hist, test_loss_hist = [], [], []
        train_acc_hist,  val_acc_hist,  test_acc_hist  = [], [], []

        # Early-stopping helpers
        no_improvement_epochs = 0
        best_epoch = 0

        # Disable eager inside graph-compiled loops
        tf.config.run_functions_eagerly(False)

        # ============================================================= #
        #                       Fine-tuning epochs                      #
        # ============================================================= #
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")
            t0 = time.time()

            # -------------------- Training -------------------- #
            epoch_train_loss = 0.0
            epoch_train_acc  = []

            for step, batch in enumerate(train_ds):
                img = tf.squeeze(batch[0][:, 0, :, :])
                aud = tf.squeeze(batch[1][:, 0, :, :])
                loss, logits = train_step(
                    model, img, aud, batch[2], batch[3], optimizer
                )
                # Convert to numpy for metric calc
                loss, logits = loss.numpy(), logits.numpy()
                acc = np.mean(
                    np.argmax(logits, axis=-1) == np.argmax(batch[3], axis=-1)
                )
                epoch_train_loss += loss
                epoch_train_acc.append(acc)

            epoch_train_loss /= len(epoch_train_acc)
            epoch_train_acc  = np.mean(epoch_train_acc)
            print("Training Step Finished.")
            print(f"Training Loss over epoch: {epoch_train_loss:.4f}")
            print(f"Training Accuracy over epoch: {epoch_train_acc*100:.2f}")

            # -------------------- Validation ------------------- #
            epoch_val_loss = 0.0
            epoch_val_acc  = []

            for batch in val_ds:
                img = tf.squeeze(batch[0][:, 0, :, :])
                aud = tf.squeeze(batch[1][:, 0, :, :])
                loss, logits = validation_step(
                    model, img, aud, batch[2], batch[3]
                )
                loss, logits = loss.numpy(), logits.numpy()
                acc = np.mean(
                    np.argmax(logits, axis=-1) == np.argmax(batch[3], axis=-1)
                )
                epoch_val_loss += loss
                epoch_val_acc.append(acc)

            epoch_val_loss /= len(epoch_val_acc)
            epoch_val_acc  = np.mean(epoch_val_acc)
            print(f"Validation Loss: {epoch_val_loss:.4f}")
            print(f"Validation Accuracy over epoch: {epoch_val_acc*100:.2f}")

            # -------------------- Testing ---------------------- #
            epoch_test_loss = 0.0
            epoch_test_acc  = []

            for batch in test_ds:
                img = tf.squeeze(batch[0][:, 0, :, :])
                aud = tf.squeeze(batch[1][:, 0, :, :])
                loss, logits = test_step(
                    model, img, aud, batch[2], batch[3]
                )
                loss, logits = loss.numpy(), logits.numpy()
                acc = np.mean(
                    np.argmax(logits, axis=-1) == np.argmax(batch[3], axis=-1)
                )
                epoch_test_loss += loss
                epoch_test_acc.append(acc)

            epoch_test_loss /= len(epoch_test_acc)
            epoch_test_acc  = np.mean(epoch_test_acc)
            print(f"Test Loss: {epoch_test_loss:.4f}")
            print(f"Test Accuracy over epoch: {epoch_test_acc*100:.2f}")
            print(f"Time taken: {time.time() - t0:.2f}s")

            # ---------------- Early-stopping logic -------------- #
            if len(val_loss_hist) == 0:
                best_epoch = epoch
                no_improvement_epochs = 0
            elif epoch_val_loss < min(val_loss_hist):
                best_epoch = epoch
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1  # no improvement this epoch

            # Persist history
            train_loss_hist.append(epoch_train_loss)
            train_acc_hist.append(epoch_train_acc)
            val_loss_hist.append(epoch_val_loss)
            val_acc_hist.append(epoch_val_acc)
            test_loss_hist.append(epoch_test_loss)
            test_acc_hist.append(epoch_test_acc)

        # ----------------------------------------------------------------- #
        # Select epoch with highest val-acc, break ties with highest test-acc
        # ----------------------------------------------------------------- #
        val_acc_arr  = np.round(np.array(val_acc_hist,  dtype=np.float32), 4)
        test_acc_arr = np.round(np.array(test_acc_hist, dtype=np.float32), 4)

        if val_acc_arr.size:
            best_idxs    = np.where(val_acc_arr == val_acc_arr.max())[0]
            best_idx     = best_idxs[np.argmax(test_acc_arr[best_idxs])]
            avg_acc      = (val_acc_arr[best_idx] + test_acc_arr[best_idx]) / 2
            print(
                f"Average-accuracy({avg_acc*100:.2f}%) ,Best test-accuracy "
                f"({test_acc_arr[best_idx]*100:.2f}%) according to max "
                f"validation-accuracy ({val_acc_arr[best_idx]*100:.2f}%) "
                f"at epoch {best_idx}."
            )
        else:
            avg_acc = 0.0
            print(
                f"Average-accuracy(0%) ,Best test-accuracy (0%) according to "
                f"max validation-accuracy (0%) at epochs."
            )

        Show_result[subject_id] = avg_acc * 100  # store as percentage
        print(f"{subject_id} = {Show_result[subject_id]}")

    # -------------------- Summary across subjects -------------------- #
    for k, v in Show_result.items():
        print(f"{k} = {v}")