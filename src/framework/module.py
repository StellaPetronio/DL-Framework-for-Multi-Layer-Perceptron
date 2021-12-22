class Module(object):
    """Module Abstract Class
    Represents an abstract layer of a Neural Network, that sustain 2 type of pass
    - the forward pass, to make a best 'guess' of the output for a given set of inputs. 
    - the backward pass, to update the weigths of the layer. 
    
    There are 2 abstract subsclasses of Module
    1. ModuleWithGrad
    2. ModuleWithoutGrad

    There are 4 instanciable subclasses of Module:
    1. Sequential
    2. Linear
    3. ReLu
    4. Tanh
    """

    def __init__(self, has_grad_update):
        """
        Parameter
        --------
        has_grad_update (bool): 'True' if the model has updatable weights to be learned during training. 
        """
        self.has_grad_update = has_grad_update

    def __call__(self, input):
        """This function allows to call the forward pass easily"""
        return self.forward(input)

    def forward(self, input):
        """
        Provided the input of the layer, returns its output

        Parameters
        ----------
        input (tensor): Tensor that will be activated. Size must match the self.size prop.
            for a matrix multiplication

        Returns
        -------
        output (tensor): forwarded input

        """
        raise NotImplementedError

    
    def backward(self, grad_wrt_output):
        """Backward pass.
        Given the gradient of the loss w.r.t. the output, computes the gradient 
        w.r.t. the input of the layer. 

        It also accumulates the gradients w.r.t the module's weights in order to be later
        updated.

        Parameters
        ----------
        grad_wrt_output (tensor): The gradient of the loss w.r.t. module's output.
        """
        raise NotImplementedError


class ModuleWithGrad(Module):
    """
    Abstract class.
    Represents a Module with a gradient that has to be updated 
    during learning. 
    """
    def __init__(self):
        super().__init__(has_grad_update = True)

    ### Functions for gradients 

    def reset_grads(self):
        """Reset all the gradients wrt to the parameters of the layer
        """
        raise NotImplementedError

    def gradient_update(self, eta, lambda_, regularization = 'none'):
        """Performs the gradient update step
        """
        raise NotImplementedError

    ### Function for the regularization

    def regularization_loss_update(self, regularization, lambda_):
        """Given the wanted regularization method, it computes the 
        loss increment which must be added to the training loss. 
        """
        raise NotImplementedError


    ### Reset function 

    def initialize_weights(self):
        """Must initialize the weights of the layer"""
        raise NotImplementedError



class ModuleWithoutGrad(Module):
    """
    Abstract class.
    Represents a Module without a gradient. 
    This is typically the case for activation layers. 
    """
    def __init__(self):
        super().__init__(has_grad_update = False)
