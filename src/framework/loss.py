class Loss(object):
    # TODO: add 'model' property to loss

    def compute_loss(self, output, y_train):
        raise NotImplementedError


class MSE(Loss):

    def compute_loss(self, output, y_train):
        """Returns a loss
        Parameters:
            output (tensor): predicted tensor
            y_train (tensor): target tensor
        """
        return (output-y_train).pow(2).sum()


    def dloss(self, output, y_train):
        """Loss derivative"""
        return 2 * (output-y_train)
