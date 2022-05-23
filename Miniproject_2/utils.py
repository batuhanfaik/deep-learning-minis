import torch

def check_inputs(inputs, length=1):
    if len(inputs) != length:
        raise TypeError(f"Expected {length} inputs, got {len(inputs)}")


def get_gradient(grads):
    if len(grads) > 0:
        check_inputs(grads)
        return grads[0]
    
    return torch.tensor(1.0)

def zeros(shape):
    x = torch.empty(shape)
    x.fill_(0.0)
    return x

def ones(shape):
    x = torch.empty(shape)
    x.fill_(1.0)
    return x

def zeros_like(tensor):
    x = torch.empty(tensor.shape)
    x.fill_(0.0)
    return x

def ones_like(tensor):
    x = torch.empty(tensor.shape)
    x.fill_(1.0)
    return x