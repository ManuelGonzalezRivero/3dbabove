# Import Packages
import numpy as np
 
# Create Data Set
def createDataSetOverlapSquares(k,m):
    """
    Create the data set used in regression
    Arguments:
        - k - The number of input features
        - m - The number of samples to generate
    Return:
        - x - The (k,m) Data
        - y - The (1,m) labels
    """
    assert(k>0 and m>0)
 
    dims = {'numExperiments':0,'numSamps':1}
    class1  = np.random.uniform(10,20,(k,m))
    class2  = np.random.uniform(5,15,(k,m))
    xTrain = np.concatenate((class1,class2),axis=1)
    yTrain = np.concatenate((np.ones((1,m)), np.zeros((1,m))),axis = 1)
 
    class1 = np.random.uniform(10,20,(k,m))
    class2 = np.random.uniform(5,15,(k,m))
    xTest = np.concatenate((class1,class2),axis=1)
    yTest = np.concatenate((np.ones((1,m)), np.zeros((1,m))),axis = 1)
 
    return xTrain,xTest,yTrain,yTest
 
# Initialize The Model
def initializeModel(k):
    """
    Initializez the weights and bias used in the model
    Arguments:
        - k - The number of input features
        - m - The number of samples to generate
    Return:
        - x - The (k,m) Data
        - y - The (1,m) labels
    """
    assert(k>0)
    weights = np.random.randn(k,1) # Initialize the weights to a random value
    bias    = 0                    # Initialize bias parameter to 0
 
    return weights,bias
 
# Define the sigmoid function
def sigmoid(z):
    """
    For arbitrary input size, takes the element wise sigmoid
    Arguments:
        - z - the input to the activation function
    Return:
        - a - value of sigmoid(z) for each element of z
    """
    a = 1/(1+np.exp(-z))
    return a
 
# compute costs
def computeCostsAndGradients(weights,bias,x,y):
    """
    compute the cost and gradient of the forward pass using x and y
    for a model containing weights and bias
    Arguments:
        - weights - the weights of the model
        - bias    - the model bias
    Return:
        - x - the x input to the model
        - y - the expected label
    """
    # For numerical Stability
    eps = (1/10**7)
 
    # Compute the model estimates recall z = w'x+b a = sigma(z)
    z = np.dot(weights.transpose(),x)+bias
 
    a = sigmoid(z)
 
    # Compute the Updates Chain Rule Intermediates
    cost = -np.mean(y*np.log(a)+(1.0-y)*np.log(1-a))
    dyhat = (-y/(a+(eps)))+(1.-y)/((1.-a)+(eps))
    dz = a*(1.-a)
    dwn = x
 
    # Compute Updates
    dw = np.mean(dyhat*dz*x,axis=1,keepdims=True)
    db = np.mean(dyhat*dz,axis=1,keepdims=True)
 
    grads = {"dw": dw,
             "db": db}
 
    return grads, cost
 
def optimize(weights,bias,xTrain,yTrain,iterations,learningRate):
    """
    Computes iterations number of optimizations
    Arguments:
        - weights - the weights of the model
        - bias    - the model bias
        - xTrain  - the training input
        - yTrain  - the training label
        - iterations - the number of iterations to run
        - learningRate - the step size for the optimization
    Return:
        - parameters - the weights and bias of the learned model
        - grads - the gradients at each step
        - costs - the cost at each step
    """
    costs = []
 
    for i in range(iterations):
        grads,cost = computeCostsAndGradients(weights,bias,xTrain,yTrain)
 
        dw = grads["dw"]
        db = grads["db"]
 
        #simple stochastic gradient descent
        weights = weights-learningRate*dw
        bias = bias-learningRate*db
 
        costs.append(cost)
 
    parameters = {"weights":weights,
                  "bias":bias}
 
    grads = {"dw":dw,
             "db":db}
 
    return parameters,grads,costs
 
def prediction(weights,bias,x):
    """
    Performs a prediction of y for input x and model defined by weights and bias
    Arguments:
        - weights - the weights of the model
        - bias    - the model bias
        - x       - the inputs we'd like to predict for
    Return:
        - a - the prediction from inputx
    """
    z = np.dot(weights.transpose(),x)+bias
    a = sigmoid(z)>.5
    return(a)
 
def model(xTrain, xTest, yTrain,yTest, iterations = 10000, learningRate = 0.01):
 
    # Initialize the weights and bias
    weights, bias = initializeModel(xTrain.shape[0])
 
    parameters,grads,costs = optimize(weights,bias,xTrain,yTrain,iterations,learningRate)
 
    weights = parameters["weights"]
    bias    = parameters["bias"]
 
    yPredTest  = prediction(weights, bias, xTest)
    yPredTrain = prediction(weights, bias, xTrain)
 
    predTestRes = yPredTest == yTest
    predTrainRes = yPredTrain == yTrain
 
    print(predTestRes.shape)
    d = {"weights":weights,
         "bias":bias,
         "predTestRes": predTestRes,
         "predTrainRes":predTrainRes}
 
    return d
 
# Create the data set
k = 5    # Two variables
m = 1000 # 4000 samples per iteration
xTrain,xTest,yTrain,yTest = createDataSetOverlapSquares(k,m)
 
# Call the model for generation and optimization
modelVar = model(xTrain, xTest, yTrain,yTest, iterations = 5000, learningRate = 0.01)
