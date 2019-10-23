import datetime
import os
import random
import time
import sys

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from helpers import *
import voc
import fcn8s

cudnn.benchmark = True

ckpt_path = 'ckpt'
exp_name = 'voc-fcn8s'

args = {
    'epoch_num': 10,
    'lr': 1e-3,
    'weight_decay': 1e-2,
    'momentum': 0.95,
    'lr_patience': 100, 
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': .2,  
    'max_images': 8000
}


def main(train_args):
    net = fcn8s.FCN8s(num_classes=voc.num_classes).cuda()
    # net = torch.load("segSavedModel.pth")

    curr_epoch = 1
    train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = MaskToTensor()
    restore_transform = standard_transforms.Compose([
        DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Resize(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = voc.VOC('train', transform=input_transform, target_transform=target_transform)
    print(train_set)
    sys.exit(1)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=6, shuffle=True)
    val_set = voc.VOC('val', transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=6, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=voc.ignore_label).cuda()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    # open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args)
        val_loss = validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)
        scheduler.step(val_loss)


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    last_cycle = time.time()
    for i, data in enumerate(train_loader):
        # print(i)
        # if i > train_args['max_images'] - 1:
        #     break
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == voc.num_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data.item(), N)

        curr_iter += 1

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        # print(vi)
        # if vi > train_args['max_images'] - 1:
        #     break
        inputs, gts = data
        N = inputs.size(0)
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data.item() / N, N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.data.squeeze_(0).cpu())
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, voc.num_classes)

    torch.save(net, "segSavedModel2.pth")
    # torch.save(net.state_dict(), "segSavedDict2.pth")

    if mean_iu > train_args['best_record']['mean_iu']:
    # if True:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc

        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])
            gt_pil = voc.colorize_mask(data[1])
            predictions_pil = voc.colorize_mask(data[2])
            if train_args['val_save_to_img_file']:
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')
    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main(args)