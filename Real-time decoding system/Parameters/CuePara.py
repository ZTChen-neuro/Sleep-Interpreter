# -*- coding: UTF-8 -*-

"""
@Project：TMR_Block
@File：CuePara.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Email: langaLinn@gmail.com
@Computer：Q58 Langa
@Date ：2024/3/28 16:47

@IDE：PyCharm
"""

# TODO: Start of the whole program
PORT = 0x4FF8
IO = "Utils/inpoutx64.dll"

CueDict = "Utils/CUE/CueList.csv"
CuePath = "Utils/CUE/data/"
CueConditionPath = "Utils/CUE/conditions/"

CueIntervalMin = 4
CueIntervalMax = 6

ConditionNumber = 30

CueState = [0, 1]
CueMarker = [90, 91]
CueMapping = ["TMR with TES", "TMR no TES"]
