# -*- coding: UTF-8 -*-

"""
@Project：TMR_Block
@File：Subject.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Computer：Q58 Langa
"""

# TODO: Start of the whole program
import os

from Utils.Style.Color import Color


class Subject:
    def __init__(self):
        # Basic Information of subject
        self.id = None
        self.name =None
        self.gender = None
        self.age =None

        # Experiment Date
        year = None
        month = None
        day = None

        # Set Volume
        self.volume = -1

        self.date = "%d_%02d_%02d" % (year, month, day)

        # LogDir: LOG/id_date_name_gender
        self.logDir = "LOG/%03d_%s_%s_%s" % (self.id, self.date, self.name, self.gender)
        self.decodeDir = self.logDir + "/Decode"
        self.creatFolder()

    def get_fileName(self):
        return self.logDir

    def creatFolder(self):
        isExists = os.path.exists(self.logDir)
        if not isExists:
            os.mkdir(self.logDir)
            os.mkdir(self.decodeDir)
            os.mkdir(self.decodeDir + "/data")

        else:
            print(Color.lightRed("%s already exists! It will be overwrote!!" % self.logDir))
            print(Color.lightRed("%s already exists! It will be overwrote!!" % self.logDir))
            print(Color.lightRed("%s already exists! It will be overwrote!!" % self.logDir))
            decide = input(Color.lightGreen("The folder is already existed, input 'YES' and press enter if you want to overwrite it: "))
            decide = decide.lower()
            if decide != "yes":
                raise NameError("Already Existed")
