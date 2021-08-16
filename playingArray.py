import numpy as np

n = 2
X = np.empty(shape=[0, n])

for i in range(5):
    for j  in range(2):
        X = np.append(X, [[i, j]], axis=0)

print(X)


print(np.size(X))
print(np.shape(X)[1])
print(X.size)

 # M = np.array(np.random.rand(4,4))
    # M = np.array(np.zeros([4,4]))
    
    # print(M) 
    
    # print(np.linalg.inv(M))
    
    
    #A = "AAA ".join(str(Topology) for x in Topology)
    #print(A)