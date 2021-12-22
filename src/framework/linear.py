from .module import ModuleWithGrad
import torch
import math

class Linear(ModuleWithGrad):
    """
    A Fully Connected Layer. 

    The weights are initialized using PyTorch's init method. 
    
    For full documentation, look at the functions' definition in the Module class.
    (it is where the parameters are defined)
    """

    def __init__(self, size):
        """
        Parameter
        ---------
        size ((int, int)): shape of the weight matrix
        """
        super().__init__()
        self.size = size
        self.initialize_weights()

    def initialize_weights(self):
        stdv = 1 / math.sqrt(self.size[0])
        self.w = torch.empty(self.size).uniform_(-stdv, stdv)
        self.b = torch.empty(self.size[0]).uniform_(-stdv, stdv)
        self.dL_dwi = torch.empty(self.size)
        self.dL_dbi = torch.empty(self.size[0])
    
    def forward(self, input):
        x = input
        si = x @ self.w.t() + self.b 
        self.x = x
        return si
    
    def backward(self, grad_wrt_output):
        # 1. backward pass
        dl_dsi = grad_wrt_output
        # last: dl_dxim1 = self.w.t() @ dl_dsi 
        dl_dxim1 = dl_dsi @ self.w

        # 2. gradients wrt parameters (for gradient update)
        # last: dl_dwi = dl_dsi.view(-1,1) @ self.x.view(1,-1)
        dl_dwi = dl_dsi.t() @ self.x
        dl_dbi = dl_dsi.sum(axis=0)
        self.dL_dwi.add_(dl_dwi)
        self.dL_dbi.add_(dl_dbi)

        # return the gradient wrt to input
        return dl_dxim1

    def reset_grads(self):
        # Reset all the gradients wrt to the parameters of the layer
        self.dL_dwi.zero_()
        self.dL_dbi.zero_()

    def gradient_update(self, eta, lambda_, regularization='none'):
        # Performs the gradient update step
        # check for the regularization type of the update performed. 
        if regularization == 'none':
            self.w = self.w - eta * self.dL_dwi
            self.b = self.b - eta * self.dL_dbi
        elif regularization == 'L1': 
            self.w = self.w - eta * self.dL_dwi - lambda_*self.w.abs().sum()
            self.b = self.b - eta * self.dL_dbi
        elif regularization == 'L2': 
            self.w = self.w - eta * self.dL_dwi - lambda_*self.w.pow(2).sum()
            self.b = self.b - eta * self.dL_dbi

    def regularization_loss_update(self, regularization, lambda_):
        # the loss update depends on the reg. method that is used. 
        loss_increment = 0
        if regularization == 'L2':
            loss_increment = self.w.norm().pow(2) * lambda_
        elif regularization == 'L1':
            loss_incremen = self.w.norm(p=1) * lambda_
        return loss_increment

