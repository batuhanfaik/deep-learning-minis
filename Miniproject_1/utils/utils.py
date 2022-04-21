import torch


def psnr(denoised_image: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
    """
    Computes the PSNR between two images.

    Args:
        denoised_image: the denoised image
        original_image: the original image

    Returns:
        the PSNR between the two images
    """
    mse = torch.mean((denoised_image - original_image) ** 2)
    eps = 1e-8
    return -10 * torch.log10(mse + eps)
