import torch
import math as m

def generate_dataset(nb=1000, process = False):

    """ Generate a train and a test set, normalize and standardize them optionally
        Parameters :
        - nb (int) : dataset size
        - process (bool): whether data will be normalized

        Returns:
            - x_tr (torch.Tensor): training input
            - y_tr (torch.Tensor): training target
            - x_te (torch.Tensor): test input
            - y_te (torch.Tensor): test target
    """
    x_tr, x_te = torch.empty(nb,2).uniform_(0,1), torch.empty(nb,2).uniform_(0,1)
    tr_labels = (x_tr.sub(0.5).pow(2).sum(1) < 1 / (2*m.pi)).type(torch.long)
    te_labels = (x_te.sub(0.5).pow(2).sum(1) < 1 / (2*m.pi)).type(torch.long)
    y_tr, y_te = torch.empty(nb,2), torch.empty(nb,2)
    y_tr[:,1], y_te[:,1] = tr_labels, te_labels
    y_tr[:,0], y_te[:,0] = 1-tr_labels, 1-te_labels

    if process:
        m_tr, std_tr = x_tr.mean(), x_tr.std()
        x_tr = x_tr.sub_(m_tr).div_(std_tr)
        x_te = x_te.sub_(m_tr).div_(std_tr)
    return x_tr, y_tr, x_te, y_te

def shuffled_data(x, y, shuffle = True):
    """
    Shuffles dataset for stochastic gradient descent
    Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): target
        - shuffle (bool): if set to true, shuffle data
    Returns:
        - shuffled_x (torch.Tensor): shuffled input
        - shuffled_y (torch.Tensor): shuffled target
    """
    n = y.shape[0]
    if shuffle:
        shuffled_ind = torch.randperm(n)
        shuffled_x = x[shuffled_ind, :]
        shuffled_y = y[shuffled_ind,:]
        return shuffled_x, shuffled_y

    return x, y

def test():
    """Utils"""
    x_tr, y_tr, x_te, y_te = generate_dataset(nb = 10, process = False)
    print("Dimensions of samples {} and of labels {}".format(x_tr.shape, y_tr.shape))
    print("Uniform: {} training samples out of 100 are classified as '1'.".format(y_tr[:,1].sum()))
    print("Should be roughly 0 if correctly normalized: {}".format(x_tr.mean(0).sum()))
    print("Should be roughly 1 if correctly standardized: {}".format(x_tr.std(0)))
    for(x,y) in zip(x_tr, y_tr):
        print("Res {} and label {} (1 if inside, 0 if outside).".format(x.sub(0.5).pow(2).sum() < 1/(2*m.pi), y))

if __name__ == "__main__":
    test()
