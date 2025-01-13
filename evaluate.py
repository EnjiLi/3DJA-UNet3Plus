import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100,
              loss_weights=[]):
    losses = {}
    # 在模型是 UNet 和 UNetPCM 时，inputs是个字典所以需要遍历
    if isinstance(inputs, dict):
        for name, x in inputs.items():
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
            if dice is True:
                dice_target = build_target(target, num_classes, ignore_index)
                loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            losses[name] = loss
    elif isinstance(inputs, tuple):
        for i, x in enumerate(inputs):
            loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
            if dice is True:
                dice_target = build_target(target, num_classes, ignore_index)
                loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            losses[f'out{i}'] = loss
        # d1,d2,d3,d4,d5 weighted average
        weights = loss_weights
        losses['out'] = 0
        for i in range(len(weights)):
            losses['out'] += weights[i] * losses[f'out{i}']
        losses['out'] = losses['out'] / sum(weights)
    else:
        losses['out'] = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)

    return losses['out']


def evaluate(model, data_loader, device, num_classes, print_freq=10, loss_weights=[0]):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs, num_classes,
                    lr_scheduler, print_freq=10, scaler=None, loss_weights=[1]):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, epochs)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255,
                             loss_weights=loss_weights)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


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
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
