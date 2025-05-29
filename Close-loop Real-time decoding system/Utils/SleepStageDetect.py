import threading
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import tensorflow as tf
import numpy as np

import time
from Model.SleepStageBlock.Model_tf.model.SIStaging import Staging
from sklearn import preprocessing

from Utils.BasicBlock import *

from Parameters import StagePara


class __SleepModel__(BasicDetectBlock):
    def __init__(self):
        super().__init__()
        self.stage = "W/N1"
        self.prob = None
        self.stageCounter = SleepStageCollector()
        self.device = StagePara.Device
        self.stable = False
        self.__interval_ori__ = StagePara.Interval
        self.__interval_factor__ = StagePara.StableFactor

        self.isStable = False
        self.isDeep = False
        self.allowIntervalChange = True
        self.deepProb = StagePara.DeepProb
        self.stageLV1 = StagePara.Stable_LV1
        self.stageLV2 = StagePara.Stable_LV2

        self.interval: float = 0.
        self.changeInterval()


    def predict(self, x):
        pass

    def changeInterval(self, custom=None, isStable=None):
        if custom is not None:
            self.interval = custom
            return
        else:
            self.stable = isStable
            if self.stable:
                self.interval = self.__interval_ori__ * self.__interval_factor__
            else:
                self.interval = self.__interval_ori__


class SleepStageCollector:
    def __init__(self):
        self.N2_count = 0  # Init N2 Stage Count
        self.N3_count = 0  # Init N3 Stage Count
        self.R_count = 0  # Init REM Stage Count
        self.NREM_count = 0  # Init N2/N3 Stage Count

        self.toString = ""

    def update(self, stage):
        count_str = None
        if stage == "N2":
            self.N2_count += 1
            self.NREM_count += 1
            count_str = "N2 Count: %d" % self.N2_count
        else:
            self.N2_count = 0

        if stage == "N3":
            self.N3_count += 1
            self.NREM_count += 1
            count_str = "N3 Count: %d" % self.N3_count
        else:
            self.N3_count = 0

        if stage == "R":
            self.R_count += 1
            self.NREM_count = 0
            count_str = "REM Count: %d" % self.R_count
        else:
            self.R_count = 0

        if stage == "W/N1":
            self.NREM_count = 0
            count_str = "Not in Sleep"

        self.toString = count_str


class SleepStageModel_tf(__SleepModel__):
    def __init__(self):
        super().__init__()
        self.basepath = "Model/SleepStageBlock/Model_tf"
        self.weights = [[1.1, 1.2, 1.0, 1.0], [1.1, 1.0, 1.2, 1.0],
                        [1.1, 1.0, 1.0, 1.1], [1.0, 1.2, 1.2, 1.0],
                        [1.0, 1.2, 1.0, 1.1], [1.0, 1.0, 1.2, 1.1]]
        self.model = self.load_model()
        self.memory = []
        self.sleepDict = ["W/N1", "N2", "N3", "R"]
        self.weight = StagePara.Weight

        self.pre_load_model()

    def load_model(self):
        model_item = Staging(fs=500, weights=self.weights)
        checkpoint_path = os.path.join(self.basepath, "checkpoint/Staging_model.ckpt")
        model_item.load_weights(checkpoint_path).expect_partial()
        return model_item

    def pre_load_model(self):
        data_now = np.random.random((3, 15000))
        self.predict(data_now)

    def predict(self, x):
        def func():
            self.detectStart = time.time()
            Result_Matrix = []
            data_now = preprocessing.scale(x, axis=1)
            # print("Data Scaling Time:", round((time.time() - start) * 1000, 3))
            start_t = time.time()

            data_now = tf.reshape(data_now, [1, 3, 15000])
            Result_Matrix1 = self.model(data_now, training=False)

            if self.prob is None:
                Result_Matrix1 = Result_Matrix1.numpy()
                stage = np.argmax(Result_Matrix1, axis=-1)
            else:
                self.prob = np.array(list(self.prob.values()), dtype=np.float32)
                Result_Matrix1 = Result_Matrix1.numpy()
                Result_Matrix1 = self.weight * Result_Matrix1 + (1 - self.weight) * self.prob
                stage = np.argmax(Result_Matrix1, axis=-1)

            start_t = time.time()
            stage = self.sleepDict[stage]
            prob_now = {}

            for i in range(len(self.sleepDict)):
                prob_now[self.sleepDict[i]] = "%.4f" % Result_Matrix1[i].item()

            self.stage = stage
            self.prob = prob_now
            self.stageCounter.update(stage)

            self.isDeep = (float(self.prob["N2"]) + float(self.prob["N3"])) >= self.deepProb
            self.isStable = self.stageCounter.NREM_count >= self.stageLV1
            if self.stage == "N2" or self.stage == "N3":
                if self.isDeep and self.stageCounter.NREM_count >= self.stageLV2 and self.allowIntervalChange:
                    self.changeInterval(isStable=True)
                else:
                    self.changeInterval(isStable=False)
            else:
                self.changeInterval(isStable=False)


            self.detectEnd = time.time()

        f = threading.Thread(target=func, args=())
        f.start()
        f.join()
