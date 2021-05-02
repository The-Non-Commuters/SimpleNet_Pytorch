# original coder : https://github.com/D-X-Y/ResNeXt-DenseNet
# added simpnet model
from __future__ import division

import os, sys, pdb, shutil, time, random, datetime
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pytorch_lightning as pl

import models
from datasets import Cifar10

from orkis.utils import convert_data_for_quaternion
import albumentations as A
from albumentations.pytorch import ToTensorV2

if not torch.cuda.is_available():
    sys.exit("CUDA is unavailable")

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# print('models : ',model_names)
parser = argparse.ArgumentParser(
    description="Trains ResNeXt on CIFAR or ImageNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("data_path", type=str, help="Path to dataset")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["cifar10", "cifar100", "imagenet", "svhn", "stl10"],
    help="Choose between Cifar10/100 and ImageNet.",
)
parser.add_argument(
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: "
    + " | ".join(model_names)
    + " (default: resnext29_8_64)",
)
# Optimization options
parser.add_argument(
    "--epochs", type=int, default=700, help="Number of epochs to train."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="The Learning Rate."
)
parser.add_argument("--momentum", type=float, default=0.90, help="Momentum.")
parser.add_argument(
    "--decay", type=float, default=0.002, help="Weight decay (L2 penalty)."
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[100, 190, 306, 390, 440, 540],
    help="Decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gammas",
    type=float,
    nargs="+",
    default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    help="LR is multiplied by gamma on schedule, number of gammas should be equal to schedule",
)
# Checkpoints
parser.add_argument(
    "--print_freq",
    default=200,
    type=int,
    metavar="N",
    help="print frequency (default: 200)",
)
parser.add_argument(
    "--save_path", type=str, default="./", help="Folder to save checkpoints and log."
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
# Acceleration
parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
# random seed
parser.add_argument("--manualSeed", type=int, help="manual seed")
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
# speeds things a bit more
cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# asd


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # used for file names, etc
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log = open(
        os.path.join(
            args.save_path, "log_seed_{0}_{1}.txt".format(args.manualSeed, time_stamp)
        ),
        "w",
    )
    print_log("save path : {}".format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace("\n", " ")), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == "cifar10":
        # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        # std = [x / 255 for x in [63.0, 62.1, 66.7]]

        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)

    elif args.dataset == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)

    # transforms
    transformNone = A.Compose([A.HorizontalFlip(),
                               A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                               A.Normalize(mean=mean, std=std),
                               ToTensorV2(),])

    transformGray = A.Compose([A.ToGray(p=1),
                               A.Normalize(mean=mean, std=std),
                               ToTensorV2(),])

    transformRGBShift = A.Compose([A.RGBShift(r_shift_limit=0,
                                              g_shift_limit=0,
                                              b_shift_limit=255,
                                              always_apply=True),
                                   A.Normalize(mean=mean, std=std),
                                   ToTensorV2(),])

    # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std),
    #         quat_trans,
    #     ]
    # )

    # test_transform = transforms.Compose(
    #     [
    #         transforms.CenterCrop(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std),
    #         quat_trans,
    #     ]
    # )

    if args.dataset == "cifar10":
        train_set = dset.CIFAR10(
            args.data_path, train=True, download=True
        )
        # test_set = dset.CIFAR10(
        #     args.data_path, train=False, transform=test_transform, download=True
        # )
        num_classes = 10
    elif args.dataset == "cifar100":
        train_set = dset.CIFAR100(
            args.data_path, train=True, download=True
        )
        # test_set = dset.CIFAR100(
        #     args.data_path, train=False, transform=test_transform, download=True
        # )
        num_classes = 100
    elif args.dataset == "imagenet":
        assert False, "Did not finish imagenet code"
    else:
        assert False, "Does not support dataset : {}".format(args.dataset)

    if args.dataset == "cifar10":
        train_set = Cifar10(data=train_set, transform=transformNone, seed=42)
    else:
        raise ValueError("Only CIFAR10 with albumentations is currently supported")

    # split off 10% of the training set into a validation set
    np.random.seed(42)
    data_len = len(train_set)
    split_alpha = 0.9
    train_len = int(split_alpha*data_len)
    val_len = data_len - train_len
    train_split, val_split = torch.utils.data.random_split(train_set, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=convert_data_for_quaternion if args.arch.startswith("q") else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=convert_data_for_quaternion if args.arch.startswith("q") else None
    )

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    # torch.save(net, 'net.pth')
    # init_net = torch.load('net.pth')
    # net.load_my_state_dict(init_net.state_dict())
    print_log("=> network :\n {}".format(net), log)

    # Main loop
    train_logger = pl.loggers.TensorBoardLogger(args.save_path)
    checkpointing = pl.callbacks.ModelCheckpoint(dirpath=args.save_path, mode="max", monitor="val_acc", filename='{epoch}-{val_acc:.2f}')
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        sync_batchnorm=True,
        #amp_level="O1",
        resume_from_checkpoint=args.resume if os.path.isfile(args.resume) else None,
        #gpus=args.ngpu,
        logger=train_logger,
        callbacks=[checkpointing]
    )
    
    trainer.fit(net, train_loader, val_loader)
    
    log.close()

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write("{}\n".format(print_string))
    log.flush()

if __name__ == "__main__":
    main()
