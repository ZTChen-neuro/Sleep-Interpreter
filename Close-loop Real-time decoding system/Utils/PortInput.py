import numpy as np
from ctypes import windll
import time
import threading
import os

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pandas as pd
import pygame
from Parameters import CuePara, MarkerPara
from Utils.Style.Color import Color
from Utils.BasicBlock import *


class __Port__(BasicStimulusBlock):
    def __init__(self):
        super().__init__()
        self.port = CuePara.PORT
        self.io = windll.LoadLibrary(CuePara.IO)

    def writePort(self, p, **kwargs):
        pass


class StageInput(__Port__):
    def __init__(self):
        super().__init__()
        self.sleepDict = MarkerPara.SleepDict

    def writePort(self, p, **kwargs):
        def func():
            self.io.DlPortWritePortUchar(self.port, self.sleepDict[p])
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, 0)

        f = threading.Thread(target=func, args=())
        f.start()
        f.join()


class AudioInput(__Port__):
    def __init__(self, condition):
        super().__init__()
        cuePath = pd.read_csv(CuePara.CueDict)
        cueIndex = np.array(cuePath["soundIndex"])
        self.cueList = np.array(cuePath["objects"])
        self.audioPaths = list(cuePath["path"])

        self.audioDict = {}
        self.audioLength = len(cueIndex)

        self.cueIntervalMin = CuePara.CueIntervalMin
        self.cueIntervalMax = CuePara.CueIntervalMax
        self.cueInterval = round(np.random.uniform(low=self.cueIntervalMin, high=self.cueIntervalMax), 2)

        for i in range(len(cueIndex)):
            self.audioDict[self.cueList[i]] = int(cueIndex[i])
        # print(self.audioDict)

        # Sound Output Init
        pygame.mixer.init()

        self.cueGroup = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
        print(Color.yellow(f"TMR give in Group1: {self.cueList[self.cueGroup[0]]}"))
        print(Color.yellow(f"TMR give in Group2: {self.cueList[self.cueGroup[1]]}"))

        self.cueType = CuePara.CueState
        self.cueMarker = CuePara.CueMarker
        self.cueMapping = CuePara.CueMapping
        self.cueState = self.cueType[0]

        # Mute Detect
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

    def writePort(self, p=None, **kwargs):
        def func():
            if p is None:
                stateCode = self.cueState
            else:
                stateCode = p
            volume = kwargs["volume"]
            stateCode = stateCode % 10
            if stateCode == 9:
                N = self.audioLength - 1
            elif stateCode == 8:
                N = np.random.randint(self.audioLength)
            else:
                # FIXME Need a better random
                N = int(np.random.choice(self.cueGroup[stateCode]))

            path = self.audioPaths[N]

            pygame.mixer.music.load(path)
            if volume > 0:
                pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            time.sleep(0.5)
            pygame.mixer.music.stop()

            if self.volume.GetMute() == 0:
                self.io.DlPortWritePortUchar(self.port, self.audioDict[self.cueList[N]])
                time.sleep(0.01)
                self.io.DlPortWritePortUchar(self.port, 0)
                print(Color.yellow("Play Sound %s" % self.cueList[N]))
            else:
                print(Color.yellow("Play Sound %s, but MUTED" % self.cueList[N]))

            self.stimEnd = time.time()

        f = threading.Thread(target=func, args=())
        f.start()
        f.join()

    def updateInterval(self):
        self.cueInterval = round(np.random.uniform(low=self.cueIntervalMin, high=self.cueIntervalMax), 2)

    def changeCueState(self):
        self.cueState = (self.cueState + 1) % len(self.cueType)
        self.io.DlPortWritePortUchar(self.port, self.cueMarker[self.cueState])
        time.sleep(0.01)
        self.io.DlPortWritePortUchar(self.port, 0)
        print(Color.yellow(f"Change Cue State {self.cueState}"))

class DecodeInput(__Port__):
    def __init__(self):
        super().__init__()
        cuePath = pd.read_csv(CuePara.CueDict)
        decodeIndex = np.array(cuePath["decodeIndex"])
        self.cueList = list(cuePath["objects"])

        self.decodeDict = {}
        self.audioLength = len(decodeIndex)

        for i in range(len(decodeIndex)):
            self.decodeDict[self.cueList[i]] = int(decodeIndex[i])

    def writePort(self, p, **kwargs):
        def func():
            self.io.DlPortWritePortUchar(self.port, self.decodeDict[p])
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, 0)

        f = threading.Thread(target=func, args=())
        f.start()
        f.join()


class TriggerInput(__Port__):
    def __init__(self):
        super().__init__()

    def writePort(self, p, **kwargs):
        def func():
            self.io.DlPortWritePortUchar(self.port, p)
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, 0)

        f = threading.Thread(target=func, args=())
        f.start()
        f.join()