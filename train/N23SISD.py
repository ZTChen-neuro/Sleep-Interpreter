import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import tensorflow.keras as K
import tensorflow_addons as tfa
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.N23SISD.sleep_encoder import sleep_encoder
    from model.N23SISD import SISD

# ---------------------------------------------------------------------------
# NREM2/3‑phase **single‑domain** decoding script (N2N3‑SISD)
# ---------------------------------------------------------------------------
#   • Loads pickled NREM‑only data containing EEG features.
#   • Uses a ensembled `sleep_encoder` network.
#   • Trains for a fixed number of epochs and checkpoints the model each epoch.
#   • Final checkpoint is stored alongside the multi‑encoder runs for consistency.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TFRecord → Tensor helpers
# ---------------------------------------------------------------------------

def _parse_function(example_proto, data_type):
    """Parse one TFRecord example into tensors suitable for the model.

    Each record contains serialized tensors with the following keys:
        • category_name         – one‑hot (15,) describing NREM micro‑state
        • feature_Sleep_EEG     – 55×200 preprocessed sleep EEG data (NREM epoch)
        • feature_id            – id vector (unused inside the model)
    """
    schema = {
        'category_name':        tf.io.FixedLenFeature([], tf.string),
        'feature_Sleep_EEG':    tf.io.FixedLenFeature([], tf.string),
        'feature_id':           tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, schema)

    # Deserialize byte‑strings to tensors
    category_name = tf.io.parse_tensor(parsed['category_name'], tf.float32)
    sleep_eeg     = tf.io.parse_tensor(parsed['feature_Sleep_EEG'], tf.float32)
    rec_id        = tf.io.parse_tensor(parsed['feature_id'], tf.float32)

    # -------------------- Reshape / transpose to expected dims --------------------
    # Sleep EEG 
    sleep_eeg = tf.transpose(tf.reshape(sleep_eeg, [55, 200]), [1, 0])      # → (200, 55)

    # Ancillary tensors
    rec_id        = tf.reshape(rec_id,        [150])  # kept for potential future use
    category_name = tf.reshape(category_name, [15])   # one‑hot (15,)

    return sleep_eeg, category_name

def get_dataset_from_tfrecords(record_path, batch_size, shuffle_buffer_size, cycle_length, data_type):
    """Create a tf.data pipeline for the given TFRecord path/glob."""
    files = tf.data.Dataset.list_files(record_path, shuffle=True)
    ds = files.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length)
    ds = ds.map(lambda ex: _parse_function(ex, data_type))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(batch_size)
    return ds

# ---------------------------------------------------------------------------
# One training step (graph‑compiled with @tf.function)
# ---------------------------------------------------------------------------
@tf.function
def train_step(model, sleep, category, optimizer):
    """Perform forward + backward pass on one mini‑batch."""
    with tf.GradientTape() as tape:
        loss, logits = model((sleep, category), training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits

if __name__ == "__main__":
    # -------------------- Reproducibility & paths --------------------
    utils.model.set_seeds(1642)

    basepath        = os.path.join(os.getcwd(), os.pardir, os.pardir)
    datapath   = os.path.join(basepath, 'data', 'Sleep Decoding', 'train_tfrecord')
    result_checkpoint       = os.path.join(basepath, os.pardir, 'results', 'Decoding checkpoint')
    os.makedirs(result_checkpoint, exist_ok=True)

    # -------------------- Hyper‑parameters --------------------
    batch_size     = 256
    shuffle_buffer_size = batch_size
    cycle_length   = 1
    ensemble_models  = 5
    data_type      = 'sleep'
    epochs         = 100

    # -------------------- Instantiate ensemble --------------------
    models = [sleep_encoder(length=200) for _ in range(ensemble_models)]

    # -------------------- TFRecord glob (NREM subset) --------------------
    nrem_dir   = os.path.join(datapath, 'N2N3')
    nrem_files = sorted(os.path.join(nrem_dir, f) for f in os.listdir(nrem_dir) if f.startswith('single_sleep'))

    # -------------------- Optimizer --------------------
    optimizer = tfa.optimizers.AdamW(learning_rate=8e-5, weight_decay=2e-5)

    # Disable eager execution inside loops for performance
    tf.config.run_functions_eagerly(False)

    # Metric histories
    train_loss_hist, train_acc_hist = [], []

    # -------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        tic = time.time()

        # -------------------- Dataset for this epoch --------------------
        train_ds = get_dataset_from_tfrecords(
            record_path=nrem_files,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cycle_length=cycle_length,
            data_type=data_type)

        epoch_loss_sum  = 0.0
        epoch_acc_list  = []

        for sleep, catetory in train_ds:

            batch_losses, batch_results = [], []
            for model in models:
                loss, result = train_step(model, sleep, catetory,  optimizer)
                batch_losses.append(loss)
                batch_results.append(result)

            loss_val   = tf.reduce_mean(batch_losses).numpy()
            logits_val = tf.reduce_mean(batch_results, axis=0).numpy()

            # Accuracy: predicted class vs. ground‑truth (argmax over 15 classes)
            acc_vec = np.argmax(logits_val, -1) == np.argmax(catetory, -1)
            epoch_acc_list.append(np.mean(acc_vec))
            epoch_loss_sum += loss_val

        # -------------------- Epoch metrics --------------------
        epoch_loss = epoch_loss_sum / len(epoch_acc_list)
        epoch_acc  = np.mean(epoch_acc_list)
        
        print('Training Step Finished.')
        print(f'Training Loss over epoch: {epoch_loss:.4f}')
        print(f'Training Accuracy over epoch: {epoch_acc*100:.2f}')
        print(f'Time taken: {time.time() - tic:.2f}s')

        # -------------------- Save checkpoints --------------------
        for i, m in enumerate(models):
            ckpt = os.path.join(result_checkpoint, f'N2N3_SISD_epoch{epoch:02d}_model{i:02d}.ckpt')
            m.save_weights(ckpt)

        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)

    # -------------------------------------------------------------------
    # Final ensemble checkpoint (last epoch weights)
    # -------------------------------------------------------------------
    ensemble = SISD()
    for i in range(ensemble_models):
        ckpt_last = os.path.join(result_checkpoint, f'N2N3_SISD_epoch{epochs-1:02d}_model{i:02d}.ckpt')
        ensemble.individual_models[i].load_weights(ckpt_last)

    ensemble_ckpt = os.path.join(result_checkpoint, 'N2N3_SISD_model.ckpt')
    ensemble.save_weights(ensemble_ckpt)
    print(f'Saved ensemble checkpoint to {ensemble_ckpt}')

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------

