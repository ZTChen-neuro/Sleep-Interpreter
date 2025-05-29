# -*- coding: UTF-8 -*-

"""
@Project：TMR_Block
@File：DataCollector.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Email: langaLinn@gmail.com
@Computer：Q58 Langa
@Date ：2024/3/22 16:01

@IDE：PyCharm 
"""

# TODO: Start of the whole program
import os
import threading
import time
import warnings
import numpy as np

import pandas as pd
import pylsl

from Utils.Filter import *
from Utils.Style.Color import Color
from Parameters import CollectorPara

warnings.filterwarnings("ignore")


class DataCollector:
    def __init__(self, subject=None):

        # Needed Channel Index Init
        self.sleep_EEG = CollectorPara.SleepEEG
        self.sleep_EOG = CollectorPara.SleepEOG
        self.sleep_EMG = CollectorPara.SleepEMG
        self.sleep_REF = CollectorPara.SleepREF
        self.so_EEG = CollectorPara.SOEEG
        self.sleep_EMGREF = CollectorPara.SleepEMGREF
        self.useChannel = CollectorPara.UseChannel
        self.fs = CollectorPara.FS
        self.warmingTime = CollectorPara.WarmingTime

        # Buffer Length Init
        self.sleepBuffer = CollectorPara.BufferSleep * self.fs
        self.sleepBuffer_reverse = CollectorPara.BufferSleep * self.fs * -1

        self.decodeBuffer = CollectorPara.BufferDecode * self.fs
        self.decodeBuffer_reverse = CollectorPara.BufferDecode * self.fs * -1

        # Unit Factor Init, The standard unit of this realtime system is "uV"
        units = CollectorPara.Units
        if units == "V":
            self.factor = 1e6
        elif units == "mV":
            self.factor = 1e3
        elif units == "uV":
            self.factor = 1
        elif units == "nV":
            self.factor = 1e-3
        else:
            raise TypeError("Unrecognized units, the unit should be V, mV, uV or nV!")

        # Sleep Bias Init
        self.bias = int(CollectorPara.BiasSleep * self.fs * -1)

        # Basic Filter Para Init
        self.b_sleep, self.a_sleep, _, _ = Filter_init(CollectorPara.FilterStageLow,
                                                       CollectorPara.FilterStageHigh,
                                                       self.fs,
                                                       CollectorPara.FilterStageOrder)

        self.b_emg, self.a_emg, _, _ = Filter_init(CollectorPara.FilterEMGLow,
                                                   CollectorPara.FilterEMGHigh,
                                                   self.fs,
                                                   CollectorPara.FilterEMGOrder)

        self.b_so, self.a_so, self.filterDelay, self.delay = Filter_init(CollectorPara.FilterSOLow,
                                                                         CollectorPara.FilterSOHigh,
                                                                         self.fs,
                                                                         CollectorPara.FilterSOOrder)

        # Subject Init
        self.subject = subject

        # Decoding Mapping Init
        mapping = CollectorPara.DecodeMapping
        if mapping is None:
            mapping = "Utils/DataCollector/mapping.csv"
        mapping = pd.read_csv(mapping)
        self.mapping = list(mapping["mapping"])

        # Data Chunk Init
        self.chunk_sleep_EEG = []
        self.chunk_sleep_EOG = []
        self.chunk_sleep_EMG = []
        self.chunk_so_EEG = []
        self.chunk_decode = []

        # Sample init time
        self.collectTime = None

        # LSL Streaming Inlet Init
        print(Color.blue("Looking for EEG streams..."))
        streams = pylsl.resolve_stream("type", "EEG")
        for s in streams:
            print(Color.blue("Find Device: %s" % s.name()))
        self.inlet = pylsl.StreamInlet(streams[0])
        lsl_time_correction = self.inlet.time_correction() * 1e6
        print(Color.blue("Time Correction: %.3fus" % lsl_time_correction))

        print(Color.blue("Sleep EEG Channel: %02d" % self.sleep_EEG))
        print(Color.blue("Sleep EOG Channel: %02d" % self.sleep_EOG))
        print(Color.blue("Sleep EMG Channel: %02d" % self.sleep_EMG))

        # Start Data Collector
        self.endCode = False
        threading.Thread(target=self.collect, args=()).start()

    def collect(self):
        print(Color.lightBlue("Data Collecting...\n"))
        self.collectTime = time.time()
        Anchor = True

        while True:
            sample, _ = self.inlet.pull_sample()
            if sample is None:
                print("No samples")

            if self.endCode:
                continue

            temp_REF = sample[self.sleep_REF]
            temp_EMGREF = 0 if self.sleep_EMGREF is None else sample[self.sleep_EMGREF]

            temp_EEG = sample[self.sleep_EEG] - temp_REF
            temp_EOG = sample[self.sleep_EOG] - temp_REF
            temp_EMG = sample[self.sleep_EMG] - temp_EMGREF

            temp_SO = sample[self.so_EEG] - temp_REF

            self.chunk_sleep_EEG.append(temp_EEG)
            self.chunk_sleep_EOG.append(temp_EOG)
            self.chunk_sleep_EMG.append(temp_EMG)
            self.chunk_so_EEG.append(temp_SO)
            self.chunk_decode.append(sample)

            if Anchor:
                np.save(os.path.join(self.subject.logDir, 'InitialAnchor.npy'), sample)
                Anchor = False

    def buffer(self):
        if len(self.chunk_sleep_EEG) > self.sleepBuffer:
            self.chunk_sleep_EEG = self.chunk_sleep_EEG[self.sleepBuffer_reverse:]
            self.chunk_sleep_EOG = self.chunk_sleep_EOG[self.sleepBuffer_reverse:]
            self.chunk_sleep_EMG = self.chunk_sleep_EMG[self.sleepBuffer_reverse:]
            self.chunk_so_EEG = self.chunk_so_EEG[self.sleepBuffer_reverse:]

        if len(self.chunk_decode) > self.decodeBuffer:
            self.chunk_decode = self.chunk_decode[self.decodeBuffer_reverse:]


    def stop(self):
        self.endCode = True

    def start(self):
        self.endCode = False

    def warming(self):
        print(Color.lightBlue("Data Warming...\n"))
        time.sleep(self.warmingTime)
