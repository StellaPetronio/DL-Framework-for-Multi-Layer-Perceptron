from .module import ModuleWithGrad
from utils import shuffled_data
import torch

class Sequential(ModuleWithGrad):
    """A Stack of different Modules (i.e. layers), that also behaves like a Module. 
    Indeed, it is 'equipped' with all the function expected from a module. 

    Therefore, two possible interpratations are possible for a Sequential.
    - It can be seen as a 'full model', in which case readers are encouraged to look at the method
        'train_model' defined in this class (and only for sequential, it is not a method inherited from 
        Module). This method shows how to train the model. 
    - Or, it can be seen as a 'reusable module' itself inserted later in another model. 
    
    For full documentation, look at the functions' definition in the Module class.
    (it is where the parameters are defined)
    """

    def __init__(self, layers, loss=None):
        """
        Parameters
        ----------
        layers ([Module]): ordered list of modules 
        loss (Loss or None): if the sequential is seen as a full model, it is 
            the Loss object used to compute the loss. If not, it is simply None
            and will not be used. 
        """
        super().__init__()
        self.layers = layers
        self.loss = loss
    
    def forward(self, X):
        # Run the pass over all the layers of the network
        X_i = X
        for layer in self.layers: 
            X_next = layer.forward(X_i)
            X_i = X_next
        # the function still returns the forwarded input, to behave like a module. 
        return X_next

    def backward(self, grad_wrt_output):
        # Apply the backward pass to each layer
        for layer in self.layers[-1::-1]:
            grad_wrt_input = layer.backward(grad_wrt_output)
            grad_wrt_output = grad_wrt_input
        # as expected for a 'module' returns the last computed gradient
        return grad_wrt_input

    def initialize_weights(self):
        # all the layers are responsible for their own weight initialization
        for layer in self.layers:
            if layer.has_grad_update:
                layer.initialize_weights()

    def reset_grads(self):
        # all the layers are responsible for their own gradient reset
        for layer in self.layers:
            if layer.has_grad_update:
                layer.reset_grads()

    def gradient_update(self, eta, lambda_, regularization = 'none'):
        # all the layers are responsible for their own gradient update
        for l in self.layers:
            if l.has_grad_update:
                # regularization is an argument of the gradient_update function, 
                # which specifies how to change the gradient.
                l.gradient_update(eta=eta, lambda_ = lambda_, regularization=regularization)

    def regularization_loss_update(self, regularization, lambda_):
        # early check (avoid unecessary loop)
        if regularization == 'none': return 0
        # all the layers are responsible for this computation
        loss_increment = 0
        for layer in self.layers:
            if layer.has_grad_update: # if layer.w is defined
                loss_increment += layer.regularization_loss_update(regularization, lambda_)
        return loss_increment


    ### Function to train the model

    def train_model(self, X_train, y_train, 
            n_epoch=50, 
            batch_size=10, 
            regularization='none', 
            lambda_ = 1e-6, 
            eta=1e-3, 
            verbose=True,
            shuffle=True,
            analyse_error=False, 
            X_test=None, 
            y_test=None):
        """Train the Sequential (if the sequential is seen as a model) using the Stochastic Gradient 
        Descent method.

        This function per-se is not required in the framework, the training can also be done manually.
        (if so, it can be done like it is implemented in this function)

        For the training, the function is the power of broadcasting to reach very fast training speed. 

        If provided correct input parameters, the function can also be used to assess the model while training. 
        By default, only the training data is required. 

        Parameters
        ---------
        - X_train (N_samples x dimension): training samples
        - y_train (N_samples x n_class): labels for each sample
        - n_epoch (int): number of epoch to be performed in the training process
        - batch_size (int): batch size to partition the input during training / evaluation
        - regularization ('none', 'L1', 'L2'): which regularization method to be used
        - lambda_ (float): regularizaion parameter
        - verbose (bool): if set to 'True', verbose will be printed
        - shuffle (bool): if set to 'True', performs stochastic minibatch gradient descent
        - analyse_error (bool): if 'True', the function returns an array of the train / test errors accross 
            each epoch. If it is set to True, X_test and y_test must be provided.
        - X_test (N_samples x dimension): testing samples
        - y_test (N_samples x n_class): labels for each sample

        Requires
        -------
        - the 'loss' property of the 'Module' self object must have been set to an object of type 'Loss'
        - if analyse_error is set to 'True', the parameters X_test and y_test must also be provided. 

        Returns
        -------
        - errors (array): list of the training error, and eventually of the testing error as well 
            (if analyse_error set to 'true')
        - losses (array): list of the training and test losses (if analyse_error is set to 'true')
        """
        if self.loss is None: 
            print("Can NOT train without a loss set to an Loss object")
            raise ValueError("No loss implemented")

        n_train = X_train.size(0) 
        errors = []
        losses = []

        for j in range(n_epoch):
            training_error = 0
            training_loss = 0

            # Mini-Bathc Stochastic Gradient Descent
            # (thanks to broadcasting, it's really easy)
            x, y = shuffled_data(X_train, y_train, shuffle=shuffle)
            for b in range(0, X_train.size(0), batch_size):
                # 0. reset the grads for the batch
                self.reset_grads()

                # 1. forward pass
                # breakpoint()
                output = self.forward(x[b:b+batch_size])

                # 2. Compute the loss
                # a. loss depending on the guess
                train_targets = y[b:b+batch_size]
                loss = self.loss.compute_loss(output, train_targets)
                training_loss += loss

                # b. loss depending on the regularization method
                loss_increment = self.regularization_loss_update(regularization, lambda_, )
                training_loss += loss_increment

                # 3. Backward pass
                dloss = self.loss.dloss(output, train_targets)
                self.backward(dloss)

                # 4. training error 
                training_error += (output.max(axis=1).indices != train_targets.max(axis=1).indices).sum().item() 

                # 5. update the parameters
                self.gradient_update(eta=eta, regularization = regularization, lambda_=lambda_)

            # check for verbose
            if verbose:
                print("Epoch {} training error = {:.3} % ; training loss = {:.6} ".format(j,
                    training_error/n_train*100, training_loss))

            # It is always convenient to print something at the end
            if j == n_epoch -1 :
                print("Epoch {} training error = {:.3} % ; training loss = {:.6} ".format(j, 
                    training_error / n_train * 100,training_loss))

            # finally. If evaluation is required, then perform it. 
            if analyse_error:
                test_error, test_loss = self.evaluate(X_test, y_test, 
                        batch_size=batch_size, regularization=regularization, lambda_=lambda_, verbose=False)
                errors.append(
                        (training_error/n_train, test_error))
                losses.append((training_loss, test_loss))
            else: 
                errors.append(training_error/n_train)
                losses.append(training_loss)


        return errors, losses

    def evaluate(self, x_test, y_test, batch_size = 10, regularization = 'none', lambda_ = 1e-6, verbose = True):
        """Evaluate model performance on test set.
        As we can't assume that the entire dataset can be saved in memory, we also performs input batches here

        Returns:
            - test error rate
            - test loss
        """
        # for x, y in zip(x_te.split(batch_size), y_te.split(batch_size)):

        loss, nb_err, n_te = 0, 0, x_test.size(0)
        for b in range(0, x_test.size(0), batch_size):
            # Compute prediction
            output = self.forward(x_test[b:b+batch_size])
            pred = output.argmax(dim=1)
            ground_truth = y_test[b:b+batch_size].argmax(dim=1)

            # Compute the loss 
            loss += self.loss.compute_loss(output, y_test[b:b+batch_size])
            # loss depending on the regularization method
            loss_increment = self.regularization_loss_update(regularization, lambda_, )
            loss += loss_increment
                            
            # Compute accuracy
            err = (pred != ground_truth).sum()
            nb_err += err

        if verbose:
            print("Test loss = {:.6}, test error rate = {:.3} %".format(loss, nb_err/n_te*100))

        # return test error and test loss
        return (nb_err/n_te).item(), loss


