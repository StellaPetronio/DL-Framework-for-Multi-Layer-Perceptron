from utils import*
from framework import MSE, Linear, Sequential, Tanh, ReLu
import torch

## Load dataset
x_tr, y_tr, x_te, y_te = generate_dataset(process = True)
shuffle = True

def run_model():
    torch.set_grad_enabled(False)

    ## Build model
    layer1 = Linear(size=(25,2))
    layer2 = Linear(size=(25,25))
    layer3 = Linear(size=(25,25))
    layer4 = Linear(size=(2,25))
    modelTanh = Sequential(layers=[layer1, Tanh(), layer2, Tanh(), layer3, Tanh(), layer4], loss = MSE())
    modelReLu = Sequential(layers=[layer1, ReLu(), layer2, ReLu(), layer3, ReLu(), layer4], loss = MSE())
    modelMixed = Sequential(layers=[layer1, ReLu(), layer2, ReLu(), layer3, ReLu(), layer4, Tanh()], loss=MSE())

    print("Running several models on the classification task")

    ## Tanh model
    print("\nTanh model training and testing...")
    modelTanh.train_model(x_tr, y_tr, verbose=False, shuffle=shuffle)
    modelTanh.evaluate(x_te, y_te)
    
    ## ReLU model
    print("\nReLU model training and testing...")
    modelReLu.initialize_weights()
    modelReLu.train_model(x_tr, y_tr, verbose=False, shuffle=shuffle)
    modelReLu.evaluate(x_te, y_te)

    ## Mixed model
    print("\nReLU hidden layer, Tanh activation layer trainind and testing...")
    modelMixed.initialize_weights()
    modelMixed.train_model(x_tr, y_tr, verbose=False, shuffle=shuffle)
    modelMixed.evaluate(x_te, y_te)

    ## ReLU and regularization L2
    print("\nReLU and L2 regularization...")
    modelReLu.initialize_weights()
    modelReLu.train_model(x_tr, y_tr, regularization='L2', verbose=False, shuffle=shuffle)
    modelReLu.evaluate(x_te, y_te, regularization='L2')
    
    ## ReLU and regularization L1
    print("\nReLU and L1 regularization..")
    modelReLu.initialize_weights()
    modelReLu.train_model(x_tr, y_tr, regularization='L1', verbose=False, shuffle=shuffle, lambda_=1e-7)
    modelReLu.evaluate(x_te,y_te, regularization='L1')


if __name__ == "__main__":

    run_model()
