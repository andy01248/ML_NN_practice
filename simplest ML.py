import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
    
training_inputs=np.array([[1,0,1,0],
        [1,0,0,1],
        [1,1,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [0,0,1,1],
        [0,1,0,1],
        [0,1,1,0]])
training_outputs=np.array([(1,1,0,0,1,1,0,0)]).T
np.random.seed(1)
weights=2*(np.random.random((4,1)))-1
print('initial weights are',weights)
# print(training_inputs)
# print(training_inputs.T)
for i in range(2000):
    outputs=sigmoid(np.dot(training_inputs,weights))
    #print('current outputs are',outputs)
    error=training_outputs-outputs
    #print('current errors are',error)
    adjustment=error*sigmoid_derivative(outputs)
    weights+=np.dot(training_inputs.T,adjustment) #Every same elements
    #weight adjustment

print('adjusted weights are',weights)
print('current outputs are',outputs)
print('current errors are',error)
new_inputs=[0,1,1,1]
new_outputs=sigmoid(np.dot(new_inputs,weights))
print('new outputs are', new_outputs)
