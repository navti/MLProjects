import torch
import torch.nn.functional as F
from tqdm import tqdm

from custom_metrics import *

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, targets = batch[0], batch[1]

            # move images and labels to correct device and type
            images, targets = images.to(device), targets.to(device)

            # predict the mask
            outputs = net(images)
            iou = custom_iou(outputs, targets)

    net.train()
    return iou