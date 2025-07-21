import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedL1Loss, self).__init__()

        self.reduction = reduction

    def forward(self, output, target):
        pixelwise_loss = torch.abs(output - target)  # L1 loss
        weights = pixelwise_loss.detach()  # Use the magnitude of the loss as weights
        weighted_loss = pixelwise_loss * weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()

        self.reduction = reduction

    def forward(self, output, target):
        pixelwise_loss = (output - target) ** 2  # MSE loss
        weights = torch.sqrt(pixelwise_loss.detach())  # Use the square root of the loss as weights
        weighted_loss = pixelwise_loss * weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss

# # Example usage
# output = torch.randn(1, 3, 256, 256)  # Example output from the network
# target = torch.randn(1, 3, 256, 256)  # Example target image

# l1_loss_fn = WeightedL1Loss()
# mse_loss_fn = WeightedMSELoss()

# l1_loss = l1_loss_fn(output, target)
# mse_loss = mse_loss_fn(output, target)

# print("Weighted L1 Loss:", l1_loss.item())
# print("Weighted MSE Loss:", mse_loss.item())
