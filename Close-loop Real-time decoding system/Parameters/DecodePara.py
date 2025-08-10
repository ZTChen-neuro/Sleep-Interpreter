"""
@Project：TMR_Block
@File：DecodePara.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Email: langaLinn@gmail.com
@Computer：Q58 Langa
@Date ：2024/3/28 17:47

@IDE：PyCharm
"""

# TODO: Start of the whole program\
from Parameters import BasicPara
BasePath = "Model/DecodingBlock"
Mode = "retrained"

Delay = 1.3
PickDelay = 3
NeedTime = 2
Start = int(BasicPara.FS * (Delay - NeedTime - PickDelay))
End = int(BasicPara.FS * (Delay - PickDelay))

SleepDict = {0: "alarm", 1: "apple", 2: "ball", 3: "book", 4: "box", 5: "chair",
             6: "kiwi", 7: "microphone", 8: "motorcycle", 9: "pepper",
             10: "sheep", 11: "shoes", 12: "strawberry", 13: "tomato", 14: "watch"}
