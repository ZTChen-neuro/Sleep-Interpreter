import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import copy as cp
import tensorflow.keras as K
import tensorflow_addons as tfa

# ---------------------------------------------------------------------------
# This script trains a sleep‑stage classifiers based on the
# SIStaging architecture. It supports two data sources:
#   1) TFRecord files produced elsewhere
#   2) Pickled dictionaries (used for the final evaluation set)
#
# The training loop:
#   • Builds six individual ensemble model whose predictions are averaged.
#   • Trains for a fixed number of epochs, tracking loss & accuracy.
#   • Saves the checkpoint of model at every epoch.
#   • After training, reloads the weights from the epoch with the best
#     validation accuracy and saves a single checkpoint.
#
# NOTE: Paths in this file assume the following folder structure
#   repo_root/
#       data/SI Staging/Train/Whole*.tfrecords
#       results/Staging checkpoint/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.SIStaging.stager.stager import stager
    from model.SIStaging.Staging import Staging


# ---------------------------------------------------------------------------
# TFRecord → Tensor conversion helpers
# ---------------------------------------------------------------------------

def _parse_function(example_proto):
    """Parse a single TFRecord example.

    Each example contains three serialized tensors:
        • category_name – one‑hot vector (4,)
        • feature_label – scalar class label
        • feature_data – preprocessed EEG data (3, 15000)
    """
    feature = {
        'category_name': tf.io.FixedLenFeature((), tf.string),
        'feature_label': tf.io.FixedLenFeature((), tf.string),
        'feature_data': tf.io.FixedLenFeature((), tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature)

    category_name = tf.io.parse_tensor(parsed['category_name'], tf.float32)
    label         = tf.io.parse_tensor(parsed['feature_label'], tf.int32)
    sleep_data    = tf.io.parse_tensor(parsed['feature_data'], tf.float32)

    # Ensure the tensors have explicit shapes for downstream layers
    sleep_data    = tf.reshape(sleep_data,    [3, 15000])
    label         = tf.reshape(label,         [1])
    category_name = tf.reshape(category_name, [4])

    return sleep_data, category_name, label


def get_dataset_from_tfrecords(
        record_path,
        batch_size,
        shuffle_buffer_size,
        cycle_length):
    """Create a tf.data pipeline from a (possibly glob) TFRecord path."""

    files = tf.data.Dataset.list_files(record_path, shuffle=True)

    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cycle_length  # parallel readers
    )
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 4)
    return dataset


# ---------------------------------------------------------------------------
# One training / inference step
# ---------------------------------------------------------------------------

def train_step(model, data, category_id, label, optimizer):
    """Single gradient‑descent step for one mini‑batch."""
    with tf.GradientTape() as tape:
        loss_value, preds = model((data, category_id, label), training=True)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, preds


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------- Hyper‑params & reproducibility --------------------
    batch_size           = 256
    shuffle_buffer_size  = batch_size
    cycle_length         = 1
    ensemble_models      = 6
    epochs               = 200

    utils.model.set_seeds(1642)        # reproducible runs

    # -------------------- Folder setup --------------------
    basepath           = os.path.join(os.getcwd(), os.pardir)
    datapath           = os.path.join(basepath, 'data', 'SI Staging')
    result_checkpoint  = os.path.join(basepath, 'results', 'Staging checkpoint')
    os.makedirs(result_checkpoint, exist_ok=True)

    # -------------------- Build ensemble --------------------
    weights = [
        [1.1, 1.2, 1.0, 1.0], [1.1, 1.0, 1.2, 1.0],
        [1.1, 1.0, 1.0, 1.1], [1.0, 1.2, 1.2, 1.0],
        [1.0, 1.2, 1.0, 1.1], [1.0, 1.0, 1.2, 1.1]
    ]
    individual_models = [stager(fs=500, weight=w) for w in weights]

    # -------------------- Data paths --------------------
    path_train_file = os.path.join(datapath, "Train")

    # Get *all* TFRecords under Train/Whole*.tfrecords
    path_train_data = sorted(os.path.join(path_train_file, f) for f in os.listdir(path_train_file) if f.startswith('Whole'))

    # -------------------- Optimizer --------------------
    optimizer = tfa.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-5
    )

    # -------------------- Metrics containers --------------------
    train_loss = []
    train_accuracy = []

    # Disable eager for performance (tf.function already used above)
    tf.config.run_functions_eagerly(False)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    no_improvement_epochs = 0
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        tic = time.time()

        # Class‑wise counters (for diagnostic output)
        correct_dict = {k: 0 for k in ["W", "N2", "N3", "R"]}
        total_dict   = {k: 0 for k in ["W", "N2", "N3", "R"]}
        idx_to_label = {0: "W", 1: "N2", 2: "N3", 3: "R"}

        # -------------------- Training --------------------
        train_dataset = get_dataset_from_tfrecords(
            record_path=path_train_data,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cycle_length=cycle_length
        )

        epoch_train_losses, epoch_train_acc = 0.0, []

        for step, (x, cat, y) in enumerate(train_dataset):
            # Ensemble: average loss & predictions across models
            losses, preds = [], []
            for model in individual_models:
                l, p = train_step(model, x, cat, tf.squeeze(y), optimizer)
                losses.append(l)
                preds.append(p)
            loss_value   = tf.reduce_mean(losses, axis=0).numpy()
            pred_logits  = tf.reduce_mean(preds,  axis=0).numpy()

            acc = np.mean(np.argmax(pred_logits, -1) == np.argmax(cat, -1))
            epoch_train_acc.append(acc)
            epoch_train_losses += loss_value

        # Aggregate metrics across all mini‑batches
        epoch_train_loss = epoch_train_losses / len(epoch_train_acc)
        epoch_train_acc  = np.mean(epoch_train_acc)
        print("Training Step Finished.")

        # -------------------- Logging --------------------
        if np.isnan(epoch_train_loss):
            print("NaN encountered – aborting training.")
            break

        print(f'Training Loss over epoch: {epoch_train_loss:.4f}')
        print(f'Training Accuracy over epoch: {epoch_train_acc*100:.2f}')
        print(f'Time taken: {time.time() - tic:.2f}s')

        # -------------------- Checkpoint each model --------------------
        for i, m in enumerate(individual_models):
            ckpt = os.path.join(result_checkpoint, f'Staging_epoch{epoch:02d}_model{i:02d}.ckpt')
            m.save_weights(ckpt)


        # -------------------- Log histories --------------------
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_acc)

    # -------------------------------------------------------------------
    # Post‑training: save ensemble
    # -------------------------------------------------------------------
    last_epoch = epochs-1

    # Re‑create ensemble wrapper and load best weights
    ensemble = Staging(fs=500, weights=weights)
    for i in range(ensemble_models):
        ckpt_best = os.path.join(result_checkpoint, f'Staging_epoch{last_epoch:02d}_model{i:02d}.ckpt')
        ensemble.individual_models[i].load_weights(ckpt_best)

    # Save unified ensemble checkpoint
    ensemble_ckpt = os.path.join(result_checkpoint, 'Staging_model.ckpt')
    ensemble.save_weights(ensemble_ckpt)
    print(f'Saved ensemble checkpoint to {ensemble_ckpt}')

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------


