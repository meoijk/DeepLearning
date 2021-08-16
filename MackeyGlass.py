import numpy as np
import math as math
import matplotlib.pyplot as plt

# a        = 0.2;     % value for a in eq (1)
# b        = 0.1;     % value for b in eq (1)
# tau      = 17;		% delay constant in eq (1)
# x0       = 1.2;		% initial condition: x(t=0)=x0
# deltat   = 0.1;	    % time step size (which coincides with the integration step)
# sample_n = 12000;	% total no. of samples, excluding the given initial condition
# interval = 1;	    % output is printed at every 'interval' time steps
# dx/dt = a*x(t-tau) / (1 + x(t-tau)^10) - b*x(t)


class MackeyGlass:
    def __init__(self,x,gama,beta,tau,neta,stepSize,timeSpan):
        self.x = x        
        self.beta = beta        
        self.gama = gama
        self.tau = tau
        self.neta = neta        
        self.stepSize = stepSize
        self.numOfSteps = timeSpan / stepSize
        self.subdivisionTau = np.floor(self.tau/self.stepSize).astype('int')
        #self.xList = []
        self.data = np.empty(shape=[0,2])        


    
    def evaluateDerivative(self,n):
        # dx/dt = beta * x(t-tau) / (1 + x(t-tau)^10) - gama*x(t)
        # Mackey-Glass equation 
        # γ = 1 , β = 2 , τ = 2 , n = 9.65 
        if n >= self.subdivisionTau:
            #x_dot = self.beta * self.xList[n - self.subdivisionTau] / (1 + math.pow(self.xList[n - self.subdivisionTau],self.neta) ) - self.gama * self.x        
            x_dot = self.beta * self.data[n - self.subdivisionTau,1] / (1 + math.pow(self.data[n - self.subdivisionTau,1],self.neta) ) - self.gama * self.x        
        else:
            x_dot = - self.gama * self.x       
        
        self.x += self.stepSize * x_dot      
        dt = self.stepSize * n          
        #dummy = np.array([n * self.stepSize, self.x])
        #self.xList.append(self.x)        
        self.data = np.append(self.data, [[dt,self.x]], axis=0)    
        
        print("Time: {0:f}\tx(t): {1:f}".format(dt, self.x))
        


    def integrateEq(self):
        numOfSteps = 0
        while numOfSteps < self.numOfSteps:                        
            self.evaluateDerivative(numOfSteps)
            numOfSteps += 1           

        

# if __name__ == "__main__":   
#     # x0, γ = 1 , β = 2 , τ = 2 , n = 9.65  
#     MG = MackeyGlass(1.2,1,2,2,9.65,1e-2,100)
#     MG.integrateEq()    
    

#     gcf = plt.figure()                
#     plt.plot(MG.data[:,0], MG.data[:,1])            
#     plt.ylabel("x(t)")
#     plt.grid()
#     gcf.set_size_inches(12, 5)
#     plt.show()  

#     #fig = plt.gcf()
#     #fig.set_size_inches(1, 10.5)
#     #fig.savefig('test2png.png', dpi=100)
    

        