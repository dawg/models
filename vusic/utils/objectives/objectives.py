import torch

__all__ = ["l2", "kl", "sparse_penalty", "l2_squared"]


def kl(y_hat, y):
    """
        kl divergence
    """
    return (
        (y * (y.add(1e-6).log() - y_hat.add(1e-6).log()) + (y_hat - y))
        .sum(dim=-1)
        .mean()
    )


def l2(y_hat, y):
    """
        l2 loss
    """
    return torch.norm(y - y_hat, 2, dim=-1).mean()


def l2_squared(weights):
    """
        Desc:
            compute l2 squared
        
        Args:
            weights (torch.tensor): matrix containing the weights of a network

    """
    return weights.pow(2.0).sum()


def sparse_penalty(weights):
    """
        Desc:
            compute sparsity penalty of a weight matrix

        Args:
            weights (torch.tensor): matrix containing the weights of a network
    """
    return weights.data.diag().abs().sum()
