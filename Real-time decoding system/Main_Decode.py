# -*- coding: UTF-8 -*-

"""
@Project：TMR_Multi

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Computer：Q58 Langa
"""

# TODO: Start of the whole program
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from Utils.DataCollector.DataCollector import DataCollector
from Utils.SleepStageDetect import SleepStageModel_tf
from Utils.PortInput import *
from Utils.SleepDecode import SleepDecodeModel

from Subject import Subject

from Utils.Style.Color import Color
from Parameters import BasicPara


class CloseLoop:
    def __init__(self):
        print("Close Loop System Start\n")
        globalStart = time.time()

        # Basic Init of all system
        print(Color.red("Now initialize basic parameters and subjects..."))
        start = time.time()
        self.fs = BasicPara.FS
        self.subject = Subject()
        self.triggerPort = TriggerInput()
        duration = round((time.time() - start) * 1000, 3)
        print(Color.lightRed("Basic parameters and subjects initialized, use time: %.3fms\n" % duration))

        # Staging Model Init
        print(Color.red("Now initialize staging model..."))
        start = time.time()
        self.useStaging = BasicPara.UseStaging
        self.stageModel = None
        if BasicPara.Model == "tf":
            self.stageModel = SleepStageModel_tf()
        else:
            self.stageModel = SleepStageModel_tf()
        self.stagePort = StageInput()
        duration = round((time.time() - start) * 1000, 3)
        print(Color.lightRed("Stage model initialized, use time: %.3fms\n" % duration))

        # Decoding Model Init, Optional
        self.useDecode = BasicPara.UseDecode
        self.decodeModel: SleepDecodeModel
        if self.useDecode:
            print(Color.red("Now initialize decode model..."))
            start = time.time()
            self.decodeModel = SleepDecodeModel()
            self.decodePort = DecodeInput()
            self.decodePred = ""
            self.decodeProb = {}
            duration = round((time.time() - start) * 1000, 3)
            print(Color.lightRed("Decode model initialized, use time: %.3fms\n" % duration))

        # TMR Block Init, Optional
        self.useTMR = BasicPara.UseTMR
        if self.useTMR:
            print(Color.red("Now initialize targeted memory reactivation block..."))
            start = time.time()
            self.soundPort = AudioInput(self.subject.id)
            duration = round((time.time() - start) * 1000, 3)
            print(Color.lightRed("TMR block initialized, use time: %.3fms\n" % duration))

        # DataCollector Init
        print(Color.red("Now initialize data collector..."))
        start = time.time()
        self.collector = DataCollector(subject=self.subject)
        duration = round((time.time() - start) * 1000, 3)
        print(Color.lightRed("Data collector initialized, use time: %.3fms\n" % duration))

        globalDuration = round((time.time() - globalStart) * 1000, 3)
        print(Color.lightRed("All blocks initialized, use time: %.3fms\n" % globalDuration))

    def detectStaging(self):
        start = time.time()
        data = self.collector.getStagingData()
        self.stageModel.predict(data)

        self.stagePort.writePort(self.stageModel.stage)
        duration = round((time.time() - start) * 1000, 3)

        outputString = ("Stage: %s, Staging Prob: %s, StagingCount: %s, %s, %s\n" %
                        (str(self.stageModel.stage),
                         str(self.stageModel.prob),
                         self.stageModel.stageCounter.toString,
                         "Staging Use Time: %.3fms" % duration,
                         time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print(outputString)


    def decode(self, startIndex, endIndex):
        start = time.time()
        data = self.collector.getDecodeData(startIndex, endIndex)
        self.decodeModel.predict(data)
        self.decodePort.writePort(self.decodeModel.pred)
        print(Color.green("Decode sound: %s" % self.decodeModel.pred))
        duration = round((time.time() - start) * 1000, 3)

    def sound(self):
        start = time.time()
        self.soundPort.writePort(self.soundPort.cueState, volume=self.subject.volume)
        self.soundPort.updateInterval()
        duration = round((time.time() - start) * 1000, 3)

    def mainLoop(self):
        self.collector.warming()
        self.stageModel.changeInterval(isStable=True)
        isCue = True
        cueStart = time.time()
        canStage = True
        canDecode = False

        while True:
            if time.time() - self.stageModel.detectEnd >= self.stageModel.interval and canStage:
                self.detectStaging()

            if self.useTMR:
                if self.stageModel.isDeep and self.stageModel.isStable and isCue:
                    if time.time() - self.soundPort.stimEnd >= self.soundPort.cueInterval:
                        self.sound()
                        canStage = False
                        canDecode = True
                    if time.time() - cueStart > 30:
                        cueStart = time.time()
                        print(Color.yellow(f"30s TMR Ended, cueState is {self.soundPort.cueState}"))
                        self.soundPort.changeCueState()

            if canDecode:
                if self.decodeModel.delay <= time.time() - self.soundPort.stimEnd <= self.decodeModel.delay + 0.1:
                    self.decode(self.decodeModel.startIndex, self.decodeModel.endIndex)
                    canStage = True
                    canDecode = False
                elif time.time() - self.soundPort.stimEnd >= self.decodeModel.delay + 1:
                    print(Color.red("Tried to decode, but failed."))
                    self.triggerPort.writePort(36)
                    canStage = True
                    canDecode = False





if __name__ == '__main__':
    closeLoop = CloseLoop()
    closeLoop.mainLoop()
