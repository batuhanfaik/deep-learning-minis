import torch
from typing import Union


def psnr(denoised_image: torch.Tensor, original_image: torch.Tensor,
         max_range: Union[int, float, torch.Tensor] = 1.0,
         device: Union[torch.device, None] = torch.device('cpu')) -> torch.Tensor:
    """
    Computes the PSNR between two images.

    Args:
        denoised_image: the denoised image
        original_image: the original image
        max_range: the maximum value of the image
        device: the device to use

    Returns:
        the PSNR between the two images
    """
    assert denoised_image.shape == original_image.shape and original_image.ndim == 4
    max_range = torch.tensor(max_range, device=device)
    mse = torch.mean((denoised_image - original_image) ** 2, dim=(1, 2, 3)).mean()
    eps = 1e-8
    return 20 * torch.log10(max_range / torch.sqrt(mse + eps))
