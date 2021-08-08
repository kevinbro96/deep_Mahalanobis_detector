from __future__ import print_function
import argparse
import torch
from torch import nn
import data_loader
import numpy as np
import models
import os
import wandb
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from lib.attacks import XAttack
import pdb
from numpy import *
parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='/gdata2/yangkw/deep_Mahalanobis_detector/adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--vae_path', default='/gdata2/yangkw/auto_aug-master/results/main_imagenet_ce100/model_epoch52.pth', help='folder to output results')
parser.add_argument('--model_path', default='/gdata2/yangkw/auto_aug-master/results/imagenet/model_epoch90.pth', help='folder to output results')
args = parser.parse_args()
wandb.init(config=args)

def main():
    print('load model: ' + args.net_type)
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'

    model = models.ResNet50(203)
    model = nn.DataParallel(model)
    save_model = torch.load(args.model_path)['state_dict']
    model.load_state_dict(save_model)
    model = nn.DataParallel(model)
    model.cuda()
    model.float()
    model.eval()

    save_model = torch.load(args.vae_path)['state_dict']
    vae = models.CVAE_imagenet(d=64, k=128)
    vae = nn.DataParallel(vae)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys())
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)
    vae.cuda()
    vae.float()
    vae.eval()
    print('load target data: ', args.dataset)

    in_transform = transforms.Compose([transforms.Resize(256), \
                                       transforms.CenterCrop(224), \
                                       transforms.ToTensor(), \
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), \
                                       ])
    _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    attack = XAttack(model, eps_max=8/255, num_iterations=100, datasets=args.dataset, norm='linf')
    x = []
    xp =[]
    xi = []
    xd = []
    xip = []
    xdp = []
    di = []
    dd = []
    d = []

    for i, (inputs, y) in enumerate(test_loader):

        inputs = inputs.cuda()
        y = y.cuda()
        adv_inputs, adv_pred = attack(inputs, y)

        inputs = Variable(inputs)
        inputs = inputs.cuda()

        adv_inputs = Variable(adv_inputs)
        adv_inputs = adv_inputs.cuda()

        inputs_i = vae(inputs)
        inputs_d = inputs - inputs_i

        adv_inputs_i = vae(adv_inputs)
        adv_inputs_d = adv_inputs - adv_inputs_i

        delta_i = adv_inputs_i - inputs_i
        delta_d = adv_inputs_d - inputs_d
        delta = adv_inputs - inputs

        inputsn = inputs.view(args.batch_size, -1).norm(dim=1, p=2)
        adv_inputsn = adv_inputs.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        inputs_in = inputs_i.view(args.batch_size, -1).norm(dim=1, p=2)#, p=float('inf'))
        inputs_dn = inputs_d.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        adv_inputs_in = adv_inputs_i.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        adv_inputs_dn = adv_inputs_d.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        delta_in = delta_i.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        delta_dn = delta_d.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()#, p=float('inf'))
        deltan = delta.view(args.batch_size, -1).norm(dim=1, p=2)#.cpu().numpy()

        print(" GX: %.2f RX: %.2f GX': %.2f RX': %.2f dG: %.2f dR: %.2f d: %.2f" % (
        inputs_in.mean().item(), inputs_dn.mean().item(), \
        adv_inputs_in.mean().item(), adv_inputs_dn.mean().item(), \
        delta_in.mean().item(), delta_dn.mean().item(), deltan.mean().item()))

        x.append(inputsn.cpu().detach().numpy())
        xp.append(adv_inputsn.cpu().detach().numpy())
        xi.append(inputs_in.cpu().detach().numpy())
        xd.append(inputs_dn.cpu().detach().numpy())
        xip.append(adv_inputs_in.cpu().detach().numpy())
        xdp.append(adv_inputs_dn.cpu().detach().numpy())
        di.append(delta_in.cpu().detach().numpy())
        dd.append(delta_dn.cpu().detach().numpy())
        d.append(deltan.cpu().detach().numpy())

        imagex = torchvision.utils.make_grid(inputs[0:8], nrow=8, padding=2, normalize=True)
        imagexh = torchvision.utils.make_grid(adv_inputs[0:8], nrow=8, padding=2, normalize=True)
        imaged = torchvision.utils.make_grid(torch.clamp(delta[0:8]*3+0.5, min=-1, max=1), nrow=8, padding=2)#, normalize=True)
        imagexi = torchvision.utils.make_grid(inputs_i[0:8], nrow=8, padding=2, normalize=True)
        imagexhi = torchvision.utils.make_grid(adv_inputs_i[0:8], nrow=8, padding=2, normalize=True)
        imagedi = torchvision.utils.make_grid(torch.clamp(delta_i[0:8]*3+0.5 ,min=-1 ,max=1), nrow=8, padding=2)#, normalize=True)
        imagexd = torchvision.utils.make_grid(inputs_d[0:8], nrow=8, padding=2, normalize=True)
        imagexhd = torchvision.utils.make_grid(adv_inputs_d[0:8], nrow=8, padding=2, normalize=True)
        imagedd = torchvision.utils.make_grid(torch.clamp(delta_d[0:8]*3+0.5, min=-1,max=1),nrow=8, padding=2)#, normalize=True)

        wandb.log({"imagex.jpg": [wandb.Image(imagex)],
                   "imagexh.jpg": [wandb.Image(imagexh)],
                   "imaged.jpg": [wandb.Image(imaged)],
                   "imagexi.jpg": [wandb.Image(imagexi)],
                   "imagexhi.jpg": [wandb.Image(imagexhi)],
                   "imagedi.jpg": [wandb.Image(imagedi)],
                   "imagexd.jpg": [wandb.Image(imagexd)],
                   "imagexhd.jpg": [wandb.Image(imagexhd)],
                   "imagedd.jpg": [wandb.Image(imagedd)]}, step=i)
        print("batch-{} label:{}".format(i,y[0:8]))
        print("batch-{} adv label:{}".format(i,adv_pred[0:8]))

    print('OVERALL')
    print("X: %.2f X': %.2f GX: %.2f RX: %.2f GX': %.2f RX': %.2f dG: %.2f dR: %.2f d: %.2f" % (
        mean(x), mean(xp),\
    mean(xi), mean(xd), \
    mean(xip), mean(xdp), \
    mean(di), mean(dd), mean(d)))

    print("X: %.2f X': %.2f GX: %.2f RX: %.2f GX': %.2f RX': %.2f dG: %.2f dR: %.2f d: %.2f" % (
        std(x), std(xp),\
    std(xi), std(xd), \
    std(xip), std(xdp), \
    std(di), std(dd), std(d)))


if __name__ == '__main__':
    main()
