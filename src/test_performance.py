from utils import*
from framework import MSE, Linear, Sequential, Tanh, ReLu
import matplotlib.pyplot as plt
import time

## Load dataset
torch.manual_seed(2021)
X_train, y_train, X_test, y_test = generate_dataset(process = True)

## Define the models and the regularization to use
layer1 = Linear(size=(25,2))
layer2 = Linear(size=(25,25))
layer3 = Linear(size=(25,25))
layer4 = Linear(size=(2,25))

modelTanh = Sequential(layers=[layer1, Tanh(), layer2, Tanh(), layer3, Tanh(), layer4], loss = MSE())
modelReLu = Sequential(layers=[layer1, ReLu(), layer2, ReLu(), layer3, ReLu(), layer4], loss = MSE())
modelMixed = Sequential(layers=[layer1, ReLu(), layer2, ReLu(), layer3, ReLu(), layer4, Tanh()], loss=MSE())
models = [(modelTanh, 'none', 'Tanh - No Reg.', 0),
        (modelReLu, 'none', 'ReLu - No Reg.',0 ),
        (modelReLu, 'L1', 'ReLu - L1', 1e-7),
        (modelReLu, 'L2', 'ReLu - L2', 1e-6),
        (modelMixed, 'none', 'ReLu - Tanh - No Reg.', 0)]

## Train Error / Test Error for best model
print(" > Train / Test errors and losses for standard model ")

start_time = time.time()
modelReLu.initialize_weights()
errors, losses = modelReLu.train_model(X_train, y_train,
        n_epoch=100,
        analyse_error=True, X_test=X_test, y_test=y_test, verbose=False, regularization='none')
print("Training time: {}s".format(time.time() - start_time))
fig, axs = plt.subplots(2)

axs[0].plot(errors)
axs[1].plot(losses)

plt.xlabel('Epochs')
axs[0].set_ylabel('Error rate in %')
axs[0].set_title('Training and Test errors for ReLU model without regularization')
axs[0].legend(['Train error', 'Test error'], frameon=False)
axs[1].set_ylabel('MSE Loss')
axs[1].set_title('Training and Test Losses for ReLU model without regularization')
axs[1].legend(['Train loss', 'Test loss'], frameon=False)
plt.show()
fig.savefig('plots/1_training_test_errors_losses_ReLU.png')

print("#"*40)

## Train and test error for the different models

print(" > Train error and loss for the different models")
fig, axs = plt.subplots(2)
for model, reg, label, lbd in models:
    model.initialize_weights()
    errors, losses = model.train_model(X_train, y_train, n_epoch=150, verbose=False, regularization = reg, lambda_=lbd) # change nb_epoch to 200 to see long run effects
    axs[0].plot(errors, label = label)
    axs[1].plot(losses, label = label)

axs[0].set_title("Training Error using different models")
axs[1].set_title("Training Loss using different models")
plt.xlabel("Epochs")
axs[0].set_ylabel("Error rate in %")
axs[1].set_ylabel("MSE Loss")
axs[0].legend(frameon=False)
axs[1].legend(frameon=False)
plt.show()
fig.savefig('plots/2_models_comparison.png')


print("#"*40)

## Comparison with PyTorch
print(" > Model comparison with Pytorch framework")

import torch
import torch.nn as nn
from torch import optim

#X_train, y_train, X_test, y_test = generate_dataset(process = True)
nb_epochs, batch_size, eta = 100, 10, 0.1

# get the errors for our model
start_time = time.time()
modelReLu.initialize_weights()
print("our model: ".format(modelReLu(X_train[0,:])))
errors, _ = modelReLu.train_model(X_train, y_train, n_epoch=nb_epochs, batch_size=batch_size, eta=eta/batch_size, regularization='none')
print("Our model training time: {} s".format(time.time() - start_time))

def torch_model():
    return nn.Sequential(nn.Linear(2,25),
                         nn.ReLU(),
                         nn.Linear(25,25),
                         nn.ReLU(),
                         nn.Linear(25,25),
                         nn.ReLU(),
                         nn.Linear(25,2))

# the goal of this is to obtain the 'errors' array containing the training error
model = torch_model()
errors_pytorch = []
criterion = nn.MSELoss()

start_time = time.time()
print("pytorch model: ".format(model(X_train[0,:])))

for e in range(nb_epochs):
    nb_error, acc_loss = 0, 0
    for b in range(0, X_train.size(0), batch_size):
        x = X_train[b:b+batch_size]
        y = y_train[b:b+batch_size]
        pred = model(x)
        loss = criterion(pred, y)
        acc_loss += loss
        nb_error += (pred.argmax(1) != y.argmax(1)).sum()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
    print("Epoch {}: loss = {:.6}, training error rate = {:.3} % ".format(e, acc_loss, nb_error / X_train.size(0)*100))
    errors_pytorch.append(nb_error / X_train.size(0))
print("Pytorch training time: {} s".format(time.time() - start_time))


# make the plot
fig = plt.figure()
plt.plot(errors, label='Our model (eta=1e-3)')
plt.plot(errors_pytorch, label='PyTorch (eta=1e-3)')
plt.title("Model (ReLu, no reg.) VS Pytorch training performances")
plt.xlabel("Epochs")
plt.ylabel("Error rate ")
plt.legend()
plt.show()
fig.savefig("plots/3_model_vs_pytorch.png")

# print accuracy on test set and time performance
print("#" * 40)
print("Evaluation on test set")

## Pytorch
pred = model(X_test)
nb_error_pytorch = (pred.argmax(1) != y_test.argmax(1)).sum()
print("Pytorch test error rate : {} %".format(nb_error_pytorch/ y_test.shape[0]*100))

## ReLU model
pred = modelReLu(X_test)
nb_error_ourModel = (pred.argmax(1) != y_test.argmax(1)).sum()
print("Our ReLU model test error rate : {} %".format(nb_error_ourModel/ y_test.shape[0]*100))





