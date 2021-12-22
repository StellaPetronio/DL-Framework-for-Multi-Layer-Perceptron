from .module import Module
import torch

class Tanh(Module):
    """Tanh Activation Layer

    This layer does not have updatable weights / gradient and therefore 
    only inherits 2 methods from Module: forward / backward
    """
    def __init__(self):
        super().__init__(has_grad_update = False)

    def forward(self, X):
        #Activation function: tanh(x)
        self.s_i = X
        return X.tanh()

    def backward(self, *grad_wrt_output):
        #Given dl/dxi, returns dl/dsi
        dl_dxi = grad_wrt_output[0]
        dl_dsi = dl_dxi * self.dsigma(self.s_i)
        return dl_dsi

    ### Helper function

    def dsigma(self, x):
        """First derivative of activation function
        """
        return x.cosh().pow(-2)

