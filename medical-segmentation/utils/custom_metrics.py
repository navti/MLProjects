import torch
import torch.nn.functional as F

__all__ = ['custom_iou', 'custom_dice']

# IoU metric

def custom_iou(predictions: torch.tensor, targets: torch.tensor):
    # inputs are of shape BxCxHxW
    assert isinstance(predictions, torch.Tensor), "Not a torch tensor"
    assert isinstance(targets, torch.Tensor), "Not a torch tensor"
    smooth = 1e-6
    n_channels = predictions.shape[1]
    predictions = F.sigmoid(predictions) if n_channels == 1 else F.softmax(predictions, dim=1)
    # intersection : element wise multiplication or logical AND
    intersection = torch.logical_and(predictions, targets).float().sum((1,2,3))
    # union : element wise sum or logical OR
    union = torch.logical_or(predictions, targets).float().sum((1,2,3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

"""
dice metric
dice_coeff = [2 * intersection(A,B)] / [sum of areas of A and B]
Intersection can be expressed as multiplication. Dice coeff is differentiable and
can be part of the loss function.
""" 
def custom_dice(predictions: torch.tensor, targets: torch.tensor):
    # inputs are of shape BxCxHxW
    # return dice loss, to be used in loss function
    assert isinstance(predictions, torch.Tensor), "Not a torch tensor"
    assert isinstance(targets, torch.Tensor), "Not a torch tensor"
    smooth = 1e-6
    n_channels = predictions.shape[1]
    predictions = F.sigmoid(predictions) if n_channels == 1 else F.softmax(predictions, dim=1)
    intersection = (2 * predictions * targets).float().sum((1,2,3))
    total = predictions.sum((1,2,3)) + targets.sum((1,2,3))
    dice_coeffs = (intersection + smooth) / (total + smooth)
    return (1 - dice_coeffs.mean()) * len(predictions)
