#PyTorch End-to-end workflow
#Things to be covered:
''' 1. Data loading and preparation
    2. PyTorch model building
    3. Model training
    4. Model predictions and evaluation
    5. Saving and loading the model
    6. Putting it all together'''

import torch
from torch import nn
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#Checking pytorch version
print(f'PyTorch version: {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Linear Regression Model
class LinearRegressionModel(nn.Module): #User defined class inherits from nn.Module class
    #nn.Module is the base class for all the neural network models
    def __init__(self): #Constructor for class LinearRegressionModel; self is the object of class LinearRegressionModel
        super().__init__()#Super Constructor calls the contructor of class nn.Module
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float32)) #Starts with a random weight & tries to adjust to the ideal value
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float32)) #Starts with random bias & tries to adjust to the ideal value
        #If requires_grad argument is set to True then it gets updated in the gradient process
        #Forward Propagation Function
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x -> variable name (Input) : datatype -> return type
            return x * self.weights + self.bias

def parameters():
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim = 1)
    y = X * weight + bias
    print(f'X: {X[: 10]}; Length: {len(X)}\ny: {y[: 10]}; Length: {len(y)}')
    return X, y

def data_split():
    data, labels = parameters()
    train_split = int(0.8 * len(data))
    X_train, y_train = data[: train_split], labels[: train_split]
    X_test, y_test = data[train_split: ], labels[train_split: ]
    print(f'Train Data: {X_train}; Length: {len(X_train)}\nTest Data: {X_test}; Length: {len(X_test)}')
    print(f'Train Labels: {y_train}; Length: {len(y_train)}\nTest Labels: {y_test}; Length: {len(y_test)}')
    return X_train, y_train, X_test, y_test

def model_pred(**kwargs):
    model = LinearRegressionModel()
    with torch.inference_mode():
        y_pred = model(kwargs['test_data'])
    return y_pred

def data_viz(**kwargs):
    '''
    Plots training, test and validation data
    :param kwargs:
    '''
    pred_data = model_pred(**kwargs)
    plt.figure(figsize = (10, 7))
    #Plotting training data in blue
    plt.scatter(kwargs['train_data'], kwargs['train_labels'], c = 'b', s = 4, label = 'Training data')
    #Plotting test data in red
    plt.scatter(kwargs['test_data'], kwargs['test_labels'], c = 'r', s = 4, label = 'Test data')
    if kwargs['predictions'] is not None:
        plt.scatter(kwargs['test_data'],kwargs['predictions'] , c = 'g', s = 4, label = 'Predictions')

    plt.scatter(kwargs['test_data'], pred_data, c='g', s=4, label='Predictions')

    plt.legend()
    plt.show()
def main():
    train_data, train_labels, test_data, test_labels = data_split()
    data_viz(train_data = train_data, train_labels = train_labels, test_data = test_data, test_labels = test_labels, predictions = None)
    torch.manual_seed(42) # Setting random seed for reproducibility
    model_0 = LinearRegressionModel()
    print(model_0)
    #Checking the model parameters
    #model.parameters() returns the parameters
    print(f'List of Parameters: {list(model_0.parameters())}')
    #model.state_dict() returns an ordered dictionary of parameter names and their values
    print(f'Parameter Details: {model_0.state_dict()}')
    #Chacking the model predictions using torch.inference_mode()
    with torch.inference_mode():
        y_pred = model_0(train_labels)
    print(f'Predicted values: {y_pred}\nTrue values: {test_labels}')
if __name__ == '__main__':
    main()

