import os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import time
import mne
from sklearn import preprocessing
from scipy import signal
sys.path.insert(0, os.path.join(os.getcwd(), 'Sleep Decoding'))
from model.N23SIMD.SIMD import SIMD as N23SIMD
from model.REMSIMD.SIMD import SIMD as REMSIMD

class SleepDecodeModel:
    def __init__(self, basepath, mode='retrained', device="cpu"):
        self.basepath = basepath
        self.mode = mode
        self.model = self.load_model()
        self.device = device
        self.sleepdict = {0:"alarm",1:"apple",2:"ball",3:"book",4:"box",5:"chair",  
                    6:"kiwi",7:"microphone",8:"motorcycle",9:"pepper",
                    10:"sheep",11:"shoes",12:"strawberry",13:"tomato",14:"watch"}
        self.memory = []
        self.pre_load_model()
        
    def load_model(self):
        model_item = N23SIMD(mode=self.mode)
        checkpoint_path = os.path.join(self.basepath, "checkpoint/N2N3_SIMD_Retrained.ckpt")
        model_item.load_weights(checkpoint_path).expect_partial()
        return model_item
    
    def pre_load_model(self):
        sleep_data = np.random.random((55, 1000))
        self.predict(sleep_data)

    
    def N2N3_preprocess(self, brain_data, sampling_rate=500):
        reference_data = np.mean(brain_data, axis=0)
        brain_data = brain_data - reference_data
        downsampled_data = signal.decimate(brain_data, 5, axis=-1)
        brain_data = downsampled_data
        return brain_data
    
    def N2N3_scale(self, brain_data, baseline_=15):
        brain_data = self.N2N3_preprocess(brain_data)
        brain_data = preprocessing.robust_scale(brain_data, axis=1,
                with_centering=True, with_scaling=True, quantile_range=(25, 75), unit_variance=False)
        clamp_range = [float(-baseline_), float(baseline_)]
        brain_data = np.where(
            brain_data < clamp_range[0],
            clamp_range[0], brain_data)
        brain_data = np.where(
            brain_data > clamp_range[1],
            clamp_range[1], brain_data)
        return brain_data
    
    def show_Result(self, Result, Result_Matrix):
        label_result = Result
        prob_result = {"W/N1":Result['prob'][0], "N2":Result['prob'][1],\
                        "N3":Result['prob'][2], "R":Result['prob'][3]}
        print(label_result)
        print(prob_result)
    
    def predict(self, sleep_data):
        sleep_data = self.N2N3_scale(sleep_data)
        sleep_data = np.transpose(sleep_data, (1,0))
        sleep_data = tf.reshape(sleep_data, [1,200,55])
        Result_Matrix1 =\
                        self.model((sleep_data), training=False)
        Result_Matrix1 = Result_Matrix1.numpy()
        sequence = list(np.argsort(-Result_Matrix1))
        for i in range(len(sequence)):
            sequence[i] = self.sleepdict[sequence[i]]
        result = np.argmax(Result_Matrix1, axis = -1)
        return self.sleepdict[result],Result_Matrix1, sequence
        