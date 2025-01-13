"""
File: evaluate.py

Author: Ibra Ndiaye
Date: 2024-01-13
"""

import torch
from utils.distributed import ConfusionMatrix, DiceCoefficient, MetricLogger

def eval_model(model, data_loader, device, num_classes, print_freq=10, loss_weights=[0]):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            if isinstance(output, dict):
                output = output['out']
            elif isinstance(output, tuple):
                weights = loss_weights
                output = output[0]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
            Adjust the learning rate
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
