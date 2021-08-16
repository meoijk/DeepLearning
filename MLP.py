#import requests
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from Data import Data
from MackeyGlass import MackeyGlass


class ActivationFuncEnum(Enum):
    Lin = 0
    ReLu = 1
    Sigm = 2
    TanH = 3 

class OptimizationMethodEnum(Enum):
    GradientDescent = 0
    GradientDescentMomentum = 1
    GradientDescentRMSProp = 2
    GradientDescentMomentumRMSPropADAM = 3
    LevenbergMarquardt = 4
    
class TrainingStrategy(Enum):
    Online = 1
    Batch = 2
    MiniBatch = 3


class Neuron:
    def __init__(self,ActivationFuncEnum):
        self.actFunc = ActivationFuncEnum
        self.v = 0
        self.y = 0
        self.y_dot = 0
        self.bias = 0
        self.delta = 0
        self.dbias = 0
        self.Vdb = 0        
        self._Vdb = 0  
        self.Sdb = 0        
        self._Sdb = 0        
        
    
    def ComputeActivationLevel(self):
        if self.actFunc == ActivationFuncEnum.Lin:
            self.y = self.v
            self.y_dot = 1

        elif self.actFunc == ActivationFuncEnum.ReLu:
            if self.v < 0:                
                self.y = 0.01 * self.v
                self.y_dot = 0.01
            else:
                self.y = self.v
                self.y_dot = 1    

        elif self.actFunc == ActivationFuncEnum.Sigm:
            self.y = 1 / (1 + np.exp(-self.v))
            self.y_dot = self.y * (1 -self.y)        

        elif self.actFunc == ActivationFuncEnum.TanH:
            ep = np.exp(self.v)
            en = np.exp(-self.v)
            self.y = (ep - en)/(ep + en)
            self.y_dot = 1 - self.y*self.y


class Layer:      
    def __init__(self,numOfNeurons,actFunc):
        self.numOfNeurons = numOfNeurons
        self.actFunc = actFunc
        self.w = []
        self.dw = []
        self.Vdw = []
        self._Vdw = []
        self.Sdw = []
        self._Sdw = []
        

        
        self.Neurons = []
        for n in range(0, self.numOfNeurons):
            self.Neurons.append(Neuron(self.actFunc))
            self.Neurons[n].bias = 2 * np.random.rand() - 1
            #print("We're on time %d" % (x))
   


class MultilayerPerceptron:
    def __init__(self, topology,optMethod,trainingStrategy):
        self.topology = topology
        self.optMethod = optMethod
        self.trainingStrategy = trainingStrategy
        self.numOfLayers = len(self.topology)
        self.learninRate = 1e-2
        self.momentum = 0.9
        self.beta1 = 0.9 #dW
        self.beta2 = 0.999 #dW^2
        self.episilon = 1e-8
        self.damping = 1e-2
        self.trainingData = []
        self.layers = []
        self.rmseGoal = 1e-6
        self.maxNumOfEpochs = 1e5
        self.rmse = 0            
        self.epochs = 0  
        self.numOfParameters = 0

        # Memory allocation
        for L in range(0,self.numOfLayers):              
            if L == self.numOfLayers-1:                
                self.layers.append(Layer(self.topology[L],ActivationFuncEnum.Lin))                
            else:
                self.layers.append(Layer(self.topology[L],ActivationFuncEnum.TanH))                
            if L > 0:
                self.numOfParameters +=  self.topology[L-1] * self.topology[L] + self.topology[L]                
                self.layers[L].w = np.zeros([self.topology[L-1],self.topology[L]])
                self.layers[L].dw = np.zeros([self.topology[L-1],self.topology[L]])
                self.layers[L].Vdw = np.zeros([self.topology[L-1],self.topology[L]])
                self.layers[L]._Vdw = np.zeros([self.topology[L-1],self.topology[L]])

                if self.optMethod == OptimizationMethodEnum.GradientDescentMomentumRMSPropADAM:
                    self.layers[L].Sdw = np.zeros([self.topology[L-1],self.topology[L]])
                    self.layers[L]._Sdw = np.zeros([self.topology[L-1],self.topology[L]])
        

    # def __iadd__(self, other):        
    #     self.num = self.num + other
    #     return self.num

    def showTopology(self):
        print(self.topology)  

    def setErrorPartialDerivativesToZero(self):
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):                
                self.layers[L].Neurons[j]._Vdb = self.layers[L].Neurons[j].Vdb
                self.layers[L].Neurons[j].Vdb = 0

                self.layers[L].Neurons[j]._Sdb = self.layers[L].Neurons[j].Sdb
                self.layers[L].Neurons[j].Sdb = 0

                self.layers[L].Neurons[j].dbias = 0                                
                
                for i in range(0,self.topology[L-1]):
                    self.layers[L]._Vdw[i,j] = self.layers[L].Vdw[i,j]
                    self.layers[L].Vdw[i,j] = 0                    

                    self.layers[L]._Sdw[i,j] = self.layers[L].Sdw[i,j]
                    self.layers[L].Sdw[i,j] = 0                    

                    self.layers[L].dw[i,j] = 0
    
    def updateSynapticWeightsAndBiases(self):        
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):                    

                if self.optMethod == OptimizationMethodEnum.GradientDescent:
                    self.layers[L].Neurons[j].bias -= self.learninRate * self.layers[L].Neurons[j].dbias       
                
                elif self.optMethod == OptimizationMethodEnum.GradientDescentMomentum:
                    self.layers[L].Neurons[j].Vdb = self.momentum * self.layers[L].Neurons[j]._Vdb + (1 - self.momentum) *  self.layers[L].Neurons[j].dbias                    
                    self.layers[L].Neurons[j].bias -= self.learninRate * self.layers[L].Neurons[j].Vdb

                elif self.optMethod == OptimizationMethodEnum.GradientDescentRMSProp:
                    # "Biases may not updated using this method"
                    self.layers[L].Neurons[j].Vdb = self.momentum * self.layers[L].Neurons[j]._Vdb + (1 - self.momentum) *  self.layers[L].Neurons[j].dbias ** 2
                    self.layers[L].Neurons[j].bias -= self.learninRate * self.layers[L].Neurons[j].dbias / np.sqrt(self.layers[L].Neurons[j].Vdb)

                elif self.optMethod == OptimizationMethodEnum.GradientDescentMomentumRMSPropADAM:                    
                    self.layers[L].Neurons[j].Vdb = self.beta1 * self.layers[L].Neurons[j]._Vdb + (1 - self.beta1) *  self.layers[L].Neurons[j].dbias 
                    self.layers[L].Neurons[j].Sdb = self.beta2 * self.layers[L].Neurons[j]._Sdb + (1 - self.beta2) *  self.layers[L].Neurons[j].dbias **2          
                    
                    self.layers[L].Neurons[j].Vdb /= (1 - np.power(self.beta1, self.epochs))
                    self.layers[L].Neurons[j].Sdb /= (1 - np.power(self.beta2, self.epochs))

                    self.layers[L].Neurons[j].bias -= self.learninRate * self.layers[L].Neurons[j].Vdb / ( np.sqrt(self.layers[L].Neurons[j].Sdb) + self.episilon )
                    
                    

                for i in range(0,self.topology[L-1]):

                    if self.optMethod == OptimizationMethodEnum.GradientDescent:
                        self.layers[L].w[i,j] -= self.learninRate * self.layers[L].dw[i,j]

                    elif self.optMethod == OptimizationMethodEnum.GradientDescentMomentum:
                        self.layers[L].Vdw[i,j] = self.momentum * self.layers[L]._Vdw[i,j] + (1-self.momentum) * self.layers[L].dw[i,j]
                        self.layers[L].w[i,j] -= self.learninRate * self.layers[L].Vdw[i,j]
                    
                    elif self.optMethod == OptimizationMethodEnum.GradientDescentRMSProp:
                        self.layers[L].Vdw[i,j] = self.momentum * self.layers[L]._Vdw[i,j] + (1-self.momentum) * self.layers[L].dw[i,j] * self.layers[L].dw[i,j]
                        self.layers[L].w[i,j] -= self.learninRate * self.layers[L].dw[i,j] / np.sqrt(self.layers[L].Vdw[i,j])

                    elif self.optMethod == OptimizationMethodEnum.GradientDescentMomentumRMSPropADAM:
                        self.layers[L].Vdw[i,j] = self.beta1 * self.layers[L]._Vdw[i,j] + (1-self.beta1) * self.layers[L].dw[i,j]
                        self.layers[L].Sdw[i,j] = self.beta2 * self.layers[L]._Sdw[i,j] + (1-self.beta2) * self.layers[L].dw[i,j] ** 2 

                        self.layers[L].Vdw[i,j] /= (1 - np.power(self.beta1, self.epochs))
                        self.layers[L].Sdw[i,j] /= (1 - np.power(self.beta2, self.epochs))

                        self.layers[L].w[i,j] -= self.learninRate * self.layers[L].Vdw[i,j] / ( np.sqrt(self.layers[L].Sdw[i,j]) + self.episilon )      
        

    def errorBackpropagation(self):        
        # Setting the partial derivaties to zero
        self.setErrorPartialDerivativesToZero()        

        self.rmse = 0
        for n in range(0,len(self.trainingData)):            
            # Online training mode
            if self.trainingStrategy == TrainingStrategy.Online or self.optMethod == OptimizationMethodEnum.LevenbergMarquardt:
                self.setErrorPartialDerivativesToZero()

            error = self.feedForward(n)            
            for k in range(0,self.topology[self.numOfLayers-1]):                
                self.rmse += error[k] * error[k]                                
            for L in range(self.numOfLayers-1,0,-1):
                # Output layer
                if L == self.numOfLayers-1:                      
                    for k in range(0,self.topology[L]):                                                
                        self.layers[L].Neurons[k].delta = error[k] * self.layers[L].Neurons[k].y_dot                        
                        self.layers[L].Neurons[k].dbias += self.layers[L].Neurons[k].delta

                        for j in range(0,self.topology[L-1]):                            
                            self.layers[L].dw[j,k] += self.layers[L].Neurons[k].delta * self.layers[L-1].Neurons[j].y
                else:
                    for j in range(0,self.topology[L]):
                        sum = 0
                        for k in range(0,self.topology[L+1]):
                            sum += self.layers[L+1].Neurons[k].delta * self.layers[L+1].w[j,k]
                        self.layers[L].Neurons[j].delta = sum * self.layers[L].Neurons[j].y_dot
                        self.layers[L].Neurons[j].dbias += self.layers[L].Neurons[j].delta
                        for i in range(0,self.topology[L-1]):                            
                            self.layers[L].dw[i,j] += self.layers[L].Neurons[j].delta * self.layers[L-1].Neurons[i].y               
            
            if self.trainingStrategy == TrainingStrategy.Online or self.optMethod == OptimizationMethodEnum.LevenbergMarquardt:
                self.updateSynapticWeightsAndBiases()
                # Jacobian
                #J = self.getJacobian()
        
        self.rmse = np.sqrt(self.rmse)/(len(self.trainingData) * self.topology[self.numOfLayers-1] )       
        print("Epochs: %d, RMSE: %f" %(self.epochs, self.rmse))      
        # WB = self.synapticWeightsBiasesToArray()
        # print(','.join(map(str, WB)))  

        

        # J = self.getJacobian()
        # print(','.join(map(str, J)))  

        if self.trainingStrategy == TrainingStrategy.Batch:
            self.updateSynapticWeightsAndBiases()

    def trainingMLP(self,numOfEpochs,rmse):        
        self.maxNumOfEpochs = numOfEpochs
        self.rmseGoal = rmse

        self.epochs = 1
        self.rmse = sys.float_info.max
        
        epochsList = []
        rmseList = []        
        while self.maxNumOfEpochs > self.epochs and self.rmse > self.rmseGoal:                                    
            self.errorBackpropagation()
            epochsList.append(self.epochs)
            rmseList.append(self.rmse)            
            self.epochs += 1
        
        plt.figure()                
        plt.plot(epochsList,rmseList)            
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.grid()
        plt.show()            
            
    
    def setLearningFactor(self,alpha):
        self.learninRate = alpha

    def setMomentum(self,momentum):
        self.momentum = momentum

    def setADAMB1(self,beta1):
        self.beta1 = beta1
    
    def setADAMB2(self,beta2):
        self.beta2 = beta2
    
    def setTrainingData(self,data):
        self.trainingData = data

    def synapticWeightsBiasesToArray(self):
        wb = []
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):
                wb.append(self.layers[L].Neurons[j].bias)
                for i in range(0,self.topology[L-1]):
                    wb.append(self.layers[L].w[i,j])
        
        return wb

    def getJacobian(self):
        wb = []
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):
                wb.append(self.layers[L].Neurons[j].dbias)
                for i in range(0,self.topology[L-1]):
                    wb.append(self.layers[L].dw[i,j])
        
        return wb



    def initializeSynapticWeightsBiases(self):                
        for L in range(1,self.numOfLayers):  
            for j in range(0,self.topology[L]):
                #self.layers[L].Neurons[j].bias = 2 * np.random.random() - 1
                self.layers[L].Neurons[j].bias = np.random.random() * np.sqrt(1/self.topology[L-1])
                for i in range(0,self.topology[L-1]):
                    #self.layers[L].w[i,j] = 2 * np.random.random() - 1
                    self.layers[L].w[i,j] = np.random.random() * np.sqrt(1/self.topology[L-1])
            
    

    def getModelParametersAsVec(self):
        WB = np.zeros([self.numOfParameters,1])        
        
        p = 0
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):
                WB[p,0] = self.layers[L].Neurons[j].bias
                p += 1
                for i in range(0,self.topology[L-1]):
                    WB[p,0] = self.layers[L].w[i,j]                                                  
                    p += 1
        return WB



    def setModelsParametersFromVec(self, WB):                
        p = 0
        for L in range(1,self.numOfLayers):
            for j in range(0,self.topology[L]):
                self.layers[L].Neurons[j].bias = WB[p,0]
                p += 1
                for i in range(0,self.topology[L-1]):
                    self.layers[L].w[i,j] = WB[p,0]                         
                    p += 1


    def initializeEducatedGuess(self, numOfGuesses):        
        rmse = sys.float_info.max
        WB = []        
        i = 0

        while i < numOfGuesses:            
            self.initializeSynapticWeightsBiases()            

            dummy = 0    
            for d in range(0,len(self.trainingData)):
                error = self.feedForward(d)
                                
                for k in range(0,self.topology[self.numOfLayers-1]):
                    dummy += error[k] * error[k]                

                if dummy < rmse:
                    rmse = dummy
                    WB = self.getModelParametersAsVec()                         
            
            i += 1

        self.setModelsParametersFromVec(WB)
        




    #  public void InitialEducatedGuess(int nOfGuesses)
    # {
    #     Matrix WB = null;
    #     float rmse = float.MaxValue;

    #     for (int g = 0; g < nOfGuesses; ++g)
    #     {
    #         InitializeSynapticWeightsAndBiases();

    #         float dummy = 0f;
    #         for (int n = 0; n < TrainingData.inpVec.Count; ++n)
    #         {
    #             float[] error;
    #             FeedForward(n, out error);

    #             for (int k = 0; k < Topology[NumberOfLayers - 1]; ++k)
    #             {
    #                 dummy += error[k] * error[k];
    #             }
    #         }

    #         if (dummy < rmse)
    #         {
    #             rmse = dummy;
    #             WB = GetCurrentSynapticWeightsAndBiases();
    #         }
    #     }

    #     SetSynapticWeightsAndBiasesFromArray(WB);
    # }
    
    
    def feedForward(self,idxData):      
        inpLayer = self.trainingData.getTrainingInpData(idxData)
        outLayer = self.trainingData.getTrainingOutData(idxData)
        error = np.zeros([self.topology[self.numOfLayers-1]])
        
        for i in range(0,self.topology[0]):
            self.layers[0].Neurons[i].y = inpLayer[i]

        for L in range(1,self.numOfLayers):            
            for j in range(0, self.topology[L]):      
                self.layers[L].Neurons[j].v = 0
                for i in range(0,self.topology[L-1]):
                    self.layers[L].Neurons[j].v += self.layers[L].w[i,j] * self.layers[L-1].Neurons[i].y
                self.layers[L].Neurons[j].v += self.layers[L].Neurons[j].bias
                self.layers[L].Neurons[j].ComputeActivationLevel()
            
                if L == self.numOfLayers-1:
                    error[j] = self.layers[L].Neurons[j].y - outLayer[j]                   

        return error

    def getOutput(self):
        out = np.zeros([self.topology[self.numOfLayers-1]])
        for k in range(0,self.topology[self.numOfLayers-1]):
            out[k] = self.layers[self.numOfLayers-1].Neurons[k].y
        return out


    def validateModel(self):
        y = []
        for n in range(0,len(self.trainingData)):
            self.feedForward(n)
            out = self.getOutput()                        
            print(out)
            y.append(out)
        
        plt.plot(self.trainingData.outputVecList, label='Data')
        plt.plot(y, linestyle='dashed', label='Predicted')

        plt.legend()
        plt.grid()
        plt.show()
        

    
    def getLayer(self, id):
        return self.layers[id]


if __name__ == "__main__":
    # Mackey-Glass equation describes a time-delayed nonlinear system that produces a chaotic result.
    # x0, γ = 1 , β = 2 , τ = 2 , n = 9.65  
    MG = MackeyGlass(1.2,1,2,2,9.65,1e-2,100)
    MG.integrateEq()    

    gcf = plt.figure()                
    plt.plot(MG.data[:,0], MG.data[:,1])            
    plt.ylabel("x(t)")
    plt.grid()
    gcf.set_size_inches(12, 5)
    plt.show()  


    trainingData = Data()
    # trainingData.addTrainingData(np.array([-1,1]),np.array([1]))
    # trainingData.addTrainingData(np.array([1,-1]),np.array([1]))
    # trainingData.addTrainingData(np.array([1,1]),np.array([-1]))
    # trainingData.addTrainingData(np.array([-1,-1]),np.array([-1]))

    # XOR
    # trainingData.addTrainingData(np.array([-1,-1]),np.array([-1]))
    # trainingData.addTrainingData(np.array([-1,1]),np.array([1]))
    # trainingData.addTrainingData(np.array([1,1]),np.array([-1]))
    # trainingData.addTrainingData(np.array([1,-1]),np.array([1]))

    # Training dataset 50%
    tData = 0.5    
    numOfDelays = 2    
    trainingDataSize = np.floor(np.shape(MG.data)[0] * tData).astype('int')
    
    inp = np.empty(shape=[0,numOfDelays])
    for i in range(0, trainingDataSize):        
        #trainingData.addTrainingData(np.array([MG.data[i,1], MG.data[i+1,1]]), np.array([MG.data[i+2,1]]))
        trainingData.addTrainingData(np.array([MG.data[i,1], MG.data[i+1,1]]), np.array([MG.data[i+2,1]]))        
           
    trainingData.printData()    
   

    Topology = np.array([2, 10, 1])
    MLP = MultilayerPerceptron(Topology,OptimizationMethodEnum.GradientDescentMomentumRMSPropADAM,TrainingStrategy.Batch)    
    MLP.setTrainingData(trainingData)        
    MLP.initializeSynapticWeightsBiases()    
    #MLP.initializeEducatedGuess(500)        
    MLP.setLearningFactor(0.001)    
    MLP.setMomentum(0.9)
    MLP.setADAMB1(0.7)
    MLP.setADAMB2(0.799)    
    
    MLP.trainingMLP(1e3,1e-5)
    MLP.validateModel()

   
    
    # for n in range(0,len(MLP.trainingData)):
    #     error = MLP.feedForward(n)
    #     print(error)

    # for i in range(0, len(Topology)):
    #     print(i)


    # MLP.showTopology()
    # print(MLP.getLayer(1).W)

    # print(MLP.feedForward(1))
    # #print(str.format("{0:F}",983983))

    # print(len(MLP.topology))
    