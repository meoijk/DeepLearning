#import requests
from enum import Enum
import numpy as np

class Data:
    def __init__(self):
        self.inputVecList = []
        self.outputVecList = []
        self._isNormalized = False
    
    def addTrainingData(self, inpVec, outVec):         
        self.inputVecList.append(inpVec)
        self.outputVecList.append(outVec)

    def getTrainingInpData(self,idx):
        return self.inputVecList[idx]

    def getTrainingOutData(self,idx):
        return self.outputVecList[idx]

    def __len__(self):
        return len(self.inputVecList)
      


    def normalizeData(self):
        self._isNormalized = True
        print("Data normalized")
    
    def printData(self):
        print(self.inputVecList)
        print(self.outputVecList)




