import os
import time
import datetime
import torch

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import MyDataset
import transforms as T
from src import  ThreeDJAUNet3Plus


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, image_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = image_size

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes, model_name, bottleneck):
    if model_name == 'ThreeDJAUNet3Plus':
        model = ThreeDJAUNet3Plus(in_channels=3, n_classes=num_classes, PCM=True, bottleneck=bottleneck)
        return model
    raise NotImplementedError (f"model {model_name} is not implemented")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    # image size
    image_size = args.image_size
    # model name
    model_name = args.model_name
    bottleneck = args.bottleneck
    # using compute_mean_std.py
    mean = (0.43526826, 0.44523221, 0.41307611)
    std = (0.20436029, 0.19237618, 0.20128716)
    nowtimestr = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file_name = model_name + "_" + nowtimestr + "_Results"
    results_file = args.data_path + '/result/' + results_file_name + '.txt'
    # Save model hyperparameters
    hyperParameter_file_name = model_name + "_" + nowtimestr + "_HyperParameter"
    hyperParameter_file = args.data_path + '/result/' + hyperParameter_file_name + '.txt'

    train_dataset = MyDataset(args.data_path,
                              train=True,
                              transforms=get_transform(train=True, image_size=image_size, mean=mean, std=std))

    val_dataset = MyDataset(args.data_path,
                            train=False,
                            transforms=get_transform(train=False, image_size=image_size, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=False,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=False,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes, model_name=model_name, bottleneck=bottleneck)
    total = sum([param.nelement() for param in model.parameters()])
    print("====================Number of parameter:%.2fM====================" % (total / 1e6))

    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'SGD-M':
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create a learning rate update strategy, here it is updated once per step (not per epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.0
    start_time = time.time()

    # Writing Hyperparameters
    with open(hyperParameter_file, "a") as f:
        f.write(f"{args}, Number of parameter={total / 1e6}M")

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,
                                        loss_weights=args.loss_weights)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes, print_freq=args.print_freq,
                                 loss_weights=args.loss_weights)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        # write into txt
        with open(results_file, "a") as f:
            # Record the train_loss, lr and validation set indicators corresponding to each epoch
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/" + args.data_path + "_" + model_name + "_" + nowtimestr + "_best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./INRIA", choices=['./WHU', './INRIA','./Massachusetts'], help="dataset root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='SGD-M', choices=['SGD-M', 'Adam'], help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=5, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--image-size", default=512, type=int, help="size fo input image ")

    # model name
    parser.add_argument("--model-name", default='ThreeDJAUNet3Plus',
                        choices=['ThreeDJAUNet3Plus'],
                        help="which model to use for training")
    parser.add_argument("--model-description", default='', )
    parser.add_argument("--bottleneck", default=True)
    parser.add_argument("--loss-weights", default=[1, 1, 1, 1, 1, 1],
                        help="d0,d1,d2,d3,d4,d5 weighted average")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
