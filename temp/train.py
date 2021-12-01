import argparse
import os
import random
import time
import warnings

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data

from temp.dataloader import Youtube8MDataset
from temp.utils import save_checkpoint, accuracy, AverageMeter, ProgressMeter

best_acc1 = 0


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1
    best_acc1 = 0

    # create model
    pretrained_msg = 'pretrained ' if args.pretrained else ''
    print('Loading {}model...'.format(pretrained_msg))
    model = resnet(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # create validation dataset and dataloader
    val_dataset = Youtube8MDataset(args, 'validate')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # create train dataset and dataloader
    train_dataset = Youtube8MDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(args.start_epoch, args.epochs + 1):
        args.lr = learning_rate_scheduler.get_lr()[0]  # set current learning rate to args

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        acc1 = 0 if epoch < args.save_after else acc1
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, is_best, result_dir=args.result_dir, filename='model_last.pth', best_filename='model_last.pth')

        # update learning rate
        learning_rate_scheduler.step()

        print()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time')
    data_time = AverageMeter('Data')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    log_path = os.path.join(args.result_dir, 'log_train.txt')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1], log_path, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # compute output
        output, _ = model(images)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        acc1 = accuracy(output, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    log_path = os.path.join(args.result_dir, 'log_train.txt')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], log_path, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # compute output
            outputs, _ = model(images)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                progress.write(i)

    # Return Top1 Accuracy
    return top1.avg


def test(args):
    args.data = os.path.expanduser(args.data)

    print('Loading model...')
    model = resnet(args)

    # load checkpoint file
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError('Invalid checkpoint: {}'.format(args.resume))

    # Data Loader
    test_dataset = Youtube8MDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Average Meter
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    log_path = os.path.join(args.result_dir, 'log_test.txt')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, top1], log_path, prefix='Test: ')

    # Switch model to evaluate mode
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # compute output
            outputs, _ = model(images)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets)
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                progress.write(i)

    return top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--num_classes', default=2, type=int, metavar='N', help='number of classes')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number')
    parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.0004, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Load pretrained model.')
    parser.add_argument('--data', default='~/data/yt8m/annotation', metavar='DIR', help='path to dataset')
    parser.add_argument('--result_dir', default='results/', metavar='DIR', help='path to results')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()

    if args.evaluate:
        test(args)
    else:
        main(args)
        args.resume = os.path.join(args.result_dir, 'model_best.pth')
        test(args)
