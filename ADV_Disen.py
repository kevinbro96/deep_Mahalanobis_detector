from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import data_loader
import numpy as np
import models
import os
import wandb
import time
from torch.autograd import Variable
import pdb
import random
import torchvision
import torchvision.transforms as transforms
import calculate_log as callog
parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--lr_schedule', type=str, required=False,
                    help='comma-separated list of epochs when learning '
                         'rate should drop')
parser.add_argument('--num_epochs', type=int, required=False,
                    help='number of epochs trained')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
parser.add_argument('--vae_path', default='./data/emb2048/model_epoch172.pth', help='folder to output results')
parser.add_argument('--analyze_type', required=True, help='adv | noise')
parser.add_argument('--log_dir', type=str, default='data/logs')
parser.add_argument('--seed', default=666, type=int, help='seed')
args = parser.parse_args()
wandb.init(config=args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
log_dir = os.path.join(args.log_dir)
if os.path.exists(log_dir):
    time.sleep(5)
else:
    os.makedirs(log_dir)

def main():
    print('load model: ' + args.net_type)
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    model = models.ResNet50(num_c=1)
    model.cuda()
    model = nn.DataParallel(model)

    vae = models.CVAE(d=32, z=2048)
    vae = nn.DataParallel(vae)
    save_model = torch.load(args.vae_path)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)
    vae.eval()

    if args.lr is None:
        args.lr = 1e-1
    if args.lr_schedule is None:
        args.lr_schedule = '20'
    if args.num_epochs is None:
        args.num_epochs = 30
    lr_drop_epochs = [int(epoch_str) for epoch_str in
                      args.lr_schedule.split(',')]
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    print('load target data: ', args.dataset)
    test_clean_data = torch.load(args.outf + 'test_clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_adv_data = torch.load(args.outf + 'test_adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_noisy_data = torch.load(args.outf + 'test_noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    testset = torch.cat((test_clean_data, test_adv_data, test_noisy_data))
    testlabel = torch.cat((torch.zeros(test_clean_data.size(0)), torch.ones(test_adv_data.size(0)), torch.zeros(test_noisy_data.size(0))))

    train_clean_data = torch.load(args.outf + 'train_clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    train_adv_data = torch.load(args.outf + 'train_adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    train_noisy_data = torch.load(args.outf + 'train_noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    trainset = torch.cat((train_clean_data, train_adv_data, train_noisy_data))
    trainlabel = torch.cat((torch.zeros(train_clean_data.size(0)), torch.ones(train_adv_data.size(0)), torch.zeros(train_noisy_data.size(0))))

    shuffle = torch.randperm(trainlabel.size(0))
    trainset = trainset[shuffle]
    trainlabel = trainlabel[shuffle]

    print('start to train: ')

    def test(epoch, testset, testlabel, model, outf):

        total = 0
        output = []

        with torch.no_grad():
            for data_index in range(int(np.floor(testset.size(0) / args.batch_size))):
                inputs= testset[total: total + args.batch_size].cuda()
                targets = testlabel[total: total + args.batch_size].cuda()
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs - vae(inputs))
                outputs = torch.sigmoid(outputs).squeeze(dim=1)
                output.append(outputs)
                total += args.batch_size
            output = torch.cat(output)

            num_samples = output.shape[0]
            l1 = open('%s/confidence_TMP_In.txt' % outf, 'w')
            l2 = open('%s/confidence_TMP_Out.txt' % outf, 'w')

            for i in range(num_samples):
                if testlabel[i] == 0:
                    l1.write("{}\n".format(-output[i]))
                else:
                    l2.write("{}\n".format(-output[i]))
            l1.close()
            l2.close()
            results = callog.metric(outf, ['TMP'])
            mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            wandb.log({'TNR': 100.*results['TMP']['TNR']})
            wandb.log({'AUROC': 100.*results['TMP']['AUROC']})
            wandb.log({'DTACC': 100. * results['TMP']['DTACC']})
            wandb.log({'AUIN': 100. * results['TMP']['AUIN']})
            wandb.log({'AUOUT': 100. * results['TMP']['AUOUT']})

    start_epoch, iteration = 0, 0
    for epoch in range(start_epoch, args.num_epochs):
        total = 0
        model.train()  # now we set the model to train mode
        lr = args.lr
        for lr_drop_epoch in lr_drop_epochs:
            if epoch >= lr_drop_epoch:
                lr *= 0.1
        print(f'START EPOCH {epoch:04d} (lr={lr:.0e})')
        for data_index in range(int(np.floor(trainset.size(0)/args.batch_size))):
            if epoch < 5 and args.lr >= 0.1:
                lr = (iteration + 1) / (5 * np.floor(trainset.size(0)/args.batch_size)) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            data = trainset[total : total + args.batch_size].cuda()
            label = trainlabel[total : total + args.batch_size].cuda()

            #labels = torch.tensor(label, dtype=torch.long)
            inputs = Variable(data, requires_grad=False)
            labels = Variable(label, requires_grad=False)

            total += args.batch_size
            iteration += 1

            optimizer.zero_grad()
            logits = model(inputs- vae(inputs))
            logits = logits.squeeze(dim=1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.round(torch.sigmoid(logits))
            correct = predicted.eq(labels.data).cpu().sum()
            accuracy = correct / inputs.size(0)
            wandb.log({'loss': loss.item()}, step=iteration)
            wandb.log({'accuracy': accuracy.item()}, step=iteration)
            print(f'ITER {iteration:06d}',
                  f'accuracy: {accuracy.item() * 100:5.1f}%',
                  f'loss: {loss.item():.2f}',
                  sep='\t')

        print(f'END EPOCH {epoch:04d}')
        if epoch % 10 == 0:
            print('BEGIN VALIDATION')
            model.eval()
            test(epoch, testset, testlabel, model, log_dir)
            checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
            torch.save(model, checkpoint_fname)

    print('BEGIN VALIDATION')
    model.eval()
    test(epoch, testset, testlabel, model, log_dir)
    checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
    torch.save(model, checkpoint_fname)

if __name__ == '__main__':
    main()