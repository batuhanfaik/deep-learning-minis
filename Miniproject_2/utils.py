import torch

def check_inputs(inputs, length=1):
    if len(inputs) != length:
        raise TypeError(f"Expected {length} inputs, got {len(inputs)}")


def get_gradient(grads):
    if len(grads) > 0:
        check_inputs(grads)
        return grads[0]
    
    return torch.tensor(1.0)