from framework import MSE, Linear, Sequential, Tanh, ReLu

import torch 
import dlc_practical_prologue as prologue
import numpy as np

#%% Load the dataset
#torch.manual_seed(2021)

n1 = 784
n2 = 50
n3 = 10

train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True, normalize = True)

#%% Example 1: create a simplistic model a train it
print("Train model 1 : ReLU hidden layers, Tanh activation layer")
layer1 = Linear(size=(n2,n1))
layer2 = Linear(size=(n3,n2))
model = Sequential(layers=[layer1, ReLu(), layer2, Tanh()], loss=MSE())
model.train_model(train_input, train_target, n_epoch=100, verbose = False, shuffle=True)
model.evaluate(test_input, test_target)

print("Train model 1 : Tanh activation and hidden layers")
layer1 = Linear(size=(n2,n1))
layer2 = Linear(size=(n3,n2))
model = Sequential(layers=[layer1, Tanh(), layer2, Tanh()], loss=MSE())
model.train_model(train_input, train_target, n_epoch=100, verbose = False)
model.evaluate(test_input, test_target)


#%% Example 2: Show modulartiy: a module can be seen as layer !! (And it's not even difficult)
print("Train model 2 : Tanh both in hidden and activation layers")
layer1 = Linear(size=(n2,n1))
layer2 = Linear(size=(n3,n2))

model1 = Sequential(layers=[layer1, Tanh()])
model2 = Sequential(layers=[layer2, Tanh()])
super_model = Sequential(layers=[model1, model2], loss=MSE())
super_model.train_model(train_input, train_target, n_epoch=100, verbose=False)
super_model.evaluate(test_input, test_target)
#%% 
