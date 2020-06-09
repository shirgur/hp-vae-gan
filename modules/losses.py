import torch
import math

__all__ = ['kl_criterion', 'kl_bern_criterion']


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.mean()


def kl_bern_criterion(x):
    KLD = torch.mul(x, torch.log(x + 1e-20) - math.log(0.5)) + torch.mul(1 - x, torch.log(1 - x + 1e-20) - math.log(1 - 0.5))
    return KLD.mean()
