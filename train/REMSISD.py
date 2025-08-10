import os,sys
import numpy as np
import pickle
import tensorflow as tf
import time
import copy as cp
from collections import Counter
import tensorflow.keras as K
import tensorflow_addons as tfa
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
    import utils
    from model.REMSISD.sleep_encoder import sleep_encoder
    from model.REMSISD import SISD

# ---------------------------------------------------------------------------
# REM‑phase **single‑domain** decoding script (REM‑SISD)
# ---------------------------------------------------------------------------
#   • Loads pickled REM‑only data containing EEG features.
#   • Uses a ensembled `sleep_encoder` network.
#   • Trains for a fixed number of epochs and checkpoints the model each epoch.
#   • Final checkpoint is stored alongside the multi‑encoder runs for consistency.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset loader (pickle → tf.data)
# ---------------------------------------------------------------------------
def load_data(path, batch_size):
    file = open(path,'rb')
    dataset = pickle.load(file)
    category, Sleep_EEG = [],[]
    for _ in range(len(dataset)):
        category.append(dataset[_]['category'])
        Sleep_EEG.append(dataset[_]['Sleep_EEG'])
    category = np.asarray(category)
    Sleep_EEG = np.asarray(Sleep_EEG)
    Sleep_EEG = np.transpose(Sleep_EEG, (0,2,1))
    dataset_length = len(category)
    dataset_ = tf.data.Dataset.from_tensor_slices((Sleep_EEG, category))
    # Shuffle and then batch the dataset.
    Dataset = dataset_.shuffle(dataset_length).batch(
        batch_size, drop_remainder=False).prefetch(2)

    return Dataset

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------- Reproducibility & paths --------------------
    utils.model.set_seeds(1642)

    basepath           = os.path.join(os.getcwd(), os.pardir)
    datapath           = os.path.join(basepath, 'data', 'SI Main Train_pickle', 'REM')
    result_checkpoint  = os.path.join(basepath, 'results', 'Decoding checkpoint')
    os.makedirs(result_checkpoint, exist_ok=True)
    
    # -------------------- Hyper‑parameters --------------------
    batch_size     = 256
    ensemble_models  = 5
    epochs         = 100
    
    # -------------------- Model & optimizer --------------------
    models = [sleep_encoder(length=200) for _ in range(ensemble_models)]
    optimizer = tfa.optimizers.AdamW(learning_rate=8e-5, weight_decay=2e-5)

    # -------------------- Dataset --------------------
    pickle_files = sorted(os.path.join(datapath, f) for f in os.listdir(datapath) if f.startswith('sleep'))
    train_dataset = load_data(pickle_files, batch_size)

    # Disable eager inside loops for performance
    tf.config.run_functions_eagerly(False)

    # -------------------- Training loop --------------------
    train_loss_hist, train_acc_hist = [], []

    # -------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        tic = time.time()

        epoch_loss_sum  = 0.0
        epoch_acc_list  = []
        
        for sleep_eeg, label in train_dataset:
            Loss_value = [] ; Result_Matrix = []
            for model in models:
                loss_value, Result_Matrix1 = train_step( \
                    model, sleep_eeg ,label, optimizer)
                Loss_value.append(loss_value) ; Result_Matrix.append(Result_Matrix1)
            loss_value = tf.reduce_mean(Loss_value, axis = 0)
            Result_Matrix1 = tf.reduce_mean(Result_Matrix, axis = 0)
            loss_value, Result_Matrix1 = loss_value.numpy(), Result_Matrix1.numpy()
            # calculate the accuracy of result and origin label
            accuracy_train_1 = np.argmax(Result_Matrix1, axis=-1) == np.argmax(label, axis=-1)
            
            accuracy_train_1 = np.sum(accuracy_train_1) / accuracy_train_1.size
            Accuracy_train.append(accuracy_train_1)
            train_losses += loss_value
        train_losses = train_losses / len(Accuracy_train)
        Accuracy_train = sum(Accuracy_train) / len(Accuracy_train)
        msg = 'Training Step Finished.'
        print(msg)

        msg = "Training Loss over epoch: %.4f" % (float(train_losses),)
        print(msg)
        msg = "Training Accuracy over epoch: %.2f" % (float(Accuracy_train*100),)
        print(msg)

        # -------------------- Save checkpoints --------------------
        for i, m in enumerate(models):
            ckpt = os.path.join(result_checkpoint, f'REM_SISD_epoch{epoch:02d}_model{i:02d}.ckpt')
            m.save_weights(ckpt)

        train_loss_hist.append(train_losses)
        train_acc_hist.append(Accuracy_train)

    # -------------------------------------------------------------------
    # Final ensemble checkpoint (last epoch weights)
    # -------------------------------------------------------------------
    ensemble = SISD()
    for i in range(ensemble_models):
        ckpt_last = os.path.join(result_checkpoint, f'REM_SISD_epoch{epochs-1:02d}_model{i:02d}.ckpt')
        ensemble.individual_models[i].load_weights(ckpt_last)

    ensemble_ckpt = os.path.join(result_checkpoint, 'REM_SISD_model.ckpt')
    ensemble.save_weights(ensemble_ckpt)
    print(f'Saved ensemble checkpoint to {ensemble_ckpt}')

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------