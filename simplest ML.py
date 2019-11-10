import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
training_inputs=[[1,0,1,0],
        [1,0,0,1],
        [1,1,0,0],
        [1,1,1,1]]
training_outputs=np.array([(1,1,0,0)]).T
print(training_inputs)
np.random.seed(1)
weights=2*(np.random.random((4,1)))-1
# print(weights)
outputs=np.dot(training_inputs,weights)
print(outputs)
cost=outputs-training_outputs
print(cost)