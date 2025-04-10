# This module provides loss functions and training utilities for the CellSeg project.
# It includes functions for Dice loss, BCE-Dice loss, focal loss, and combined loss,
# as well as a custom training epoch class with learning rate scheduler steps.

import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def dice_loss(pred, target, smooth=1.):
    """
    Computes the Dice loss between predictions and targets.
    Args:
        pred: Predicted segmentation map.
        target: Ground truth segmentation map.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Mean Dice loss.
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()

    loss = (1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)))

    return loss.mean()


class BCEDiceLoss:
    __name__ = 'bce_dice'

    def __init__(self, bce_weight=0.5):
        # Initialize BCE-Dice loss with given weight.
        self.bce_weight = bce_weight
        self.device = 'cpu'
        if self.bce_weight == 0:
            self.__call__ = self.dice_loss_calc
        elif self.bce_weight == 1:
            self.__call__ = self.bce_loss_calc

    def __call__(self, pred, target):
        # Combine BCE and Dice losses weighted by bce_weight.
        return (self.bce_loss_calc(pred, target) * self.bce_weight +
                self.dice_loss_calc(pred, target) * (1 - self.bce_weight))

    def dice_loss_calc(self, pred, target):
        # Calculate Dice loss after applying sigmoid on predictions.
        return dice_loss(torch.sigmoid(pred), target).to(self.device)

    def bce_loss_calc(self, pred, target):
        # Calculate Binary Cross Entropy loss with logits.
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean').to(self.device)

    def to(self, device):
        self.device = device


class WeightedBCEDiceLoss:
    __name__ = 'bce_dice'

    def __init__(self, bce_weight=0.5, boundary_weight=0.99):
        # Initialize weighted BCE-Dice loss, giving extra weight to boundary predictions.
        self.bce_weight = bce_weight
        self.device = 'cpu'
        self.boundary_weight = boundary_weight
        if self.bce_weight == 0:
            self.__call__ = self.dice_loss_calc
        elif self.bce_weight == 1:
            self.__call__ = self.bce_loss_calc

    def __call__(self, pred, target):
        # Compute losses for cells and boundaries separately and combine them.
        bce_cells = self.bce_loss_calc(pred[:, :-1, :, :], target[:, :-1, :, :])
        bce_boundaries = self.bce_loss_calc(pred[:, -1, :, :], target[:, -1, :, :])

        dice_cells = self.dice_loss_calc(torch.sigmoid(pred[:, :-1, :, :]), target[:, :-1, :, :])
        dice_boundaries = self.dice_loss_calc(torch.sigmoid(pred[:, -1, :, :]), target[:, -1, :, :])

        # Apply different weights for boundary components.
        bce_loss_val = (bce_cells + self.boundary_weight * bce_boundaries) / (1 + self.boundary_weight)
        dice_loss_val = (dice_cells + self.boundary_weight * dice_boundaries) / (1 + self.boundary_weight)

        return bce_loss_val * self.bce_weight + dice_loss_val * (1 - self.bce_weight)

    def dice_loss_calc(self, pred, target):
        return dice_loss(torch.sigmoid(pred), target).to(self.device)

    def bce_loss_calc(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean').to(self.device)

    def to(self, device):
        self.device = device


class FocalLoss(torch.nn.Module):
    __name__ = 'focal'

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, targets):
        # Apply BCE with logits and compute focal loss.
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, targets)
        pt = torch.exp(-BCE_loss)  # pt is the probability of the prediction
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

    def to(self, device):
        self.device = device


class FocalLossMultiClass(torch.nn.Module):
    __name__ = 'focal'

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, targets):
        # Compute Cross Entropy loss and then apply focal weighting.
        CE_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

    def to(self, device):
        self.device = device


class CombinedLoss(torch.nn.Module):
    __name__ = 'bce_dice_focal'

    def __init__(self, bce_weight=0.5, alpha=0.5, gamma=2.0):
        super(CombinedLoss, self).__init__()
        # Combine focal and BCE-Dice losses.
        self.focal_loss = FocalLossMultiClass(alpha=alpha, gamma=gamma)
        self.dice_loss = BCEDiceLoss(bce_weight=bce_weight)

    def __call__(self, inputs, targets):
        focal_loss_val = self.focal_loss(inputs, targets)
        dice_loss_val = self.dice_loss(inputs, targets)
        return focal_loss_val + dice_loss_val

    def to(self, device):
        self.device = device


class TrainEpochSchedulerStep(smp.utils.train.Epoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler, scheduler_step_every_batch=False,
                 device="cpu", verbose=True):
        # Custom training epoch class that supports stepping the scheduler every batch.
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_every_batch = scheduler_step_every_batch

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        # Perform one batch update: forward pass, loss calculation, backward pass, optimizer and scheduler step.
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler_step_every_batch:
            self.scheduler.step()
        return loss, prediction

    def _format_logs(self, logs):
        # Format log output.
        str_logs = ["{} - {:7.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def run(self, dataloader):
        # Run one epoch over the dataloader.
        self.on_epoch_start()

        logs = {}
        loss_meter = smp.utils.train.AverageValueMeter()
        metrics_meters = {metric.__name__: smp.utils.train.AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not self.verbose,
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # Update loss meter and logs.
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # Update metrics for this batch.
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                lr_logs = {'LR': self.scheduler.get_last_lr()[0]}
                logs.update(lr_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
