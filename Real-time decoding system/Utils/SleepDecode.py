import os
import numpy as np
import tensorflow as tf
import threading
from sklearn import preprocessing
from scipy import signal

from Model.DecodingBlock.model.N23SIMD.SIMD import SIMD as N23SIMD
from Model.DecodingBlock.model.REMSIMD.SIMD import SIMD as REMSIMD
from Utils.BasicBlock import *

from Parameters import DecodePara


class SleepDecodeModel(BasicDetectBlock):
    def __init__(self):
        super().__init__()
        self.basePath = DecodePara.BasePath
        self.mode = DecodePara.Mode
        self.sleepDict = DecodePara.SleepDict

        self.model = self.load_model()

        self.memory = []
        self.pre_load_model()

        self.delay = DecodePara.PickDelay
        self.startIndex = DecodePara.Start
        self.endIndex = DecodePara.End

        self.pred = None
        self.prob = None

    def load_model(self):
        model_item = N23SIMD(mode=self.mode)
        checkpoint_path = os.path.join(self.basepath, "checkpoint/N2N3_SIMD_Retrained.ckpt")
        model_item.load_weights(checkpoint_path).expect_partial()
        return model_item

    def pre_load_model(self):
        sleep_data = np.random.random((55, 1000))
        self.predict(sleep_data)

    def N2N3_preprocess(self, brain_data):
        reference_data = np.mean(brain_data, axis=0)
        brain_data = brain_data - reference_data
        downsampled_data = signal.decimate(brain_data, 5, axis=-1)
        brain_data = downsampled_data
        return brain_data

    def N2N3_scale(self, brain_data, baseline_=15):
        brain_data = self.N2N3_preprocess(brain_data)
        brain_data = preprocessing.robust_scale(brain_data, axis=1,
                                                with_centering=True, with_scaling=True, quantile_range=(25, 75),
                                                unit_variance=False)
        clamp_range = [float(-baseline_), float(baseline_)]
        brain_data = np.where(
            brain_data < clamp_range[0],
            clamp_range[0], brain_data)
        brain_data = np.where(
            brain_data > clamp_range[1],
            clamp_range[1], brain_data)
        return brain_data

    def predict(self, sleep_data):
        # Data: Cue offset: -0.2 to 1.8s, that is Cue onset: 0.3 to 2.3s.
        def func(sleep_data=sleep_data):
            sleep_data = self.N2N3_scale(sleep_data)
            sleep_data = np.transpose(sleep_data, (1, 0))
            sleep_data = tf.reshape(sleep_data, [1, 200, 55])
            Result_Matrix1 =\
                        self.model((sleep_data), training=False)
            Result_Matrix1 = Result_Matrix1.numpy()

            result = np.argmax(Result_Matrix1, axis=-1)
            self.pred = self.sleepDict[result]
            self.prob = Result_Matrix1
        f = threading.Thread(target=func, args=())
        f.start()
        f.join()
        