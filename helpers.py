import torch

def replace_zeros(tensor: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    return torch.where(tensor == 0, torch.tensor(epsilon, device=tensor.device), tensor)
