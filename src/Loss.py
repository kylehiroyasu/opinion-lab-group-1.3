import torch as t

def kcl(output, target):
    """ Calculates the KCL loss for a given network output and its targets.
    Arguments:
        output {torch.Tensor[N x output_dim]} -- the output of the model
        target {torch.Tensor[N x 1]} -- the target for a given training example
    Returns:
        {torch.Tensor[1]} -- returns KCL loss
    """
    pass

def mcl(output, target):
    """ Calculates the MCL loss for a given network output and its targets.
    Arguments:
        output {torch.Tensor[N x output_dim]} -- the output of the model
        target {torch.Tensor[N x 1]} -- the target for a given training example
    Returns:
        {torch.Tensor[1]} -- returns MCL loss
    """
    pass