from .module import ModuleWithoutGrad
import torch

class ReLu(ModuleWithoutGrad):
    """ReLu Activation Layer

    This layer does not have updatable weights / gradient and therefore 
    only inherits 2 methods from Module: forward / backward
    """
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Activation function: ReLu
        self.s_i = X
        return X.clamp(min=0)

    def backward(self, grad_wrt_output):
        # Given dl/dxi, returns dl/dsi
        dl_dxi = grad_wrt_output
        dl_dsi = dl_dxi * self.dsigma(self.s_i)
        return dl_dsi

    ### Helper method 

    def dsigma(self, x):
        """First derivative of activation function
        """
        return (x>0).type(torch.int)
