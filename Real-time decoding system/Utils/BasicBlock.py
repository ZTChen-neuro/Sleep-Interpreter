# -*- coding: UTF-8 -*-

"""
@Project：TMR_Multi
@File：BasicBlock.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Email: langaLinn@gmail.com
@Computer：Q58 Langa
@Date ：2024/5/13 21:11

@IDE：PyCharm 
"""

# TODO: Start of the whole program
import time


class BasicDetectBlock(object):
    def __init__(self):
        self.detectStart = time.time()
        self.detectEnd = time.time()


class BasicStimulusBlock(object):
    def __init__(self):
        self.stimStart = time.time()
        self.stimEnd = time.time()

