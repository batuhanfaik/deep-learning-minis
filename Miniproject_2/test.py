
import torch

from modules import MaxPool2d

from torch import autograd

from modules import MSE

x = torch.rand((3, 4))
y = torch.rand((3, 4))
torch_loss = torch.nn.MSELoss()
loss = MSE()
torch_out = torch_loss(x, y)
out = loss(x, y)
print(torch_out)
print(out)
# self.assertTrue(torch.allclose(torch_out, out))
# self.assertEqual(torch_out.shape, out.shape)
# self.assertTrue(isinstance(out, GTensor))
# self.assertTrue(torch.equal(out.get_inputs()[0], x))
# self.assertTrue(torch.equal(out.get_inputs()[1], y))