import torch


class Model:
    def __init__(self) -> None:
        raise NotImplementedError

    def load_pretrained_model(self) -> None:
        raise NotImplementedError

    def train(self, train_input, train_target) -> None:
        raise NotImplementedError

    def predict(self, test_input) -> torch.Tensor:
        raise NotImplementedError
