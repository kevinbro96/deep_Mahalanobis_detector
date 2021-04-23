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

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
parser.add_argument('--vae_path', default='./data/emb2048/model_epoch172.pth', help='folder to output results')
parser.add_argument('--analyze_type', required=True, help='adv | noise')
args = parser.parse_args()
wandb.init(config=args)

def main():
    print('load model: ' + args.net_type)
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    model = models.ResNet50(num_c=args.num_classes)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
    vae = models.CVAE(d=32, z=2048)
    vae = nn.DataParallel(vae)
    save_model = torch.load(args.vae_path)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)


    print('load target data: ', args.dataset)
    test_clean_data = torch.load(args.outf + 'test_clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_adv_data = torch.load(args.outf + 'test_adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_noisy_data = torch.load(args.outf + 'test_noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))

    total = 0

    xi = []
    xd = []
    xip = []
    xdp = []
    di = []
    dd = []
    d = []

    for data_index in range(int(np.floor(test_clean_data.size(0)/args.batch_size))):
        data = test_clean_data[total : total + args.batch_size].cuda()
        adv_data = test_adv_data[total : total + args.batch_size].cuda()
        noisy_data = test_noisy_data[total : total + args.batch_size].cuda()

        total += args.batch_size
        inputs = Variable(data)

        if args.analyze_type == 'adv':
            adv_inputs = Variable(adv_data)
        else:
            adv_inputs = Variable(noisy_data)

        inputs_i = vae(inputs)
        inputs_d = inputs - inputs_i

        adv_inputs_i = vae(adv_inputs)
        adv_inputs_d = adv_inputs - adv_inputs_i

        delta_i = adv_inputs_i - inputs_i
        delta_d = adv_inputs_d - inputs_d
        delta = adv_inputs - inputs

        inputs_in = inputs_i.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        inputs_dn = inputs_d.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        adv_inputs_in = adv_inputs_i.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        adv_inputs_dn = adv_inputs_d.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        delta_in = delta_i.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        delta_dn = delta_d.view(args.batch_size, -1).norm(dim=1)#, p=float('inf'))
        delta = delta.view(args.batch_size, -1).norm(dim=1)

        print("xi: %.2f xd: %.2f xi': %.2f xd': %.2f di: %.2f dd: %.2f d: %.2f" % (
        inputs_in.mean().item(), inputs_dn.mean().item(), \
        adv_inputs_in.mean().item(), adv_inputs_dn.mean().item(), \
        delta_in.mean().item(), delta_dn.mean().item(), delta.mean().item()))

        xi.append(inputs_in)
        xd.append(inputs_dn)
        xip.append(adv_inputs_in)
        xdp.append(adv_inputs_dn)
        di.append(delta_in)
        dd.append(delta_dn)
        d.append(delta)

        successful_attacks = []
        successful_attacks.append(torch.cat([
            torchvision.utils.make_grid(inputs[0:8], padding=0),
            torchvision.utils.make_grid(adv_inputs[0:8], padding=0),
            torchvision.utils.make_grid(torch.clamp((adv_inputs[0:8] - inputs[0:8]) * 3 + 0.5, 0, 1), padding=0),
            torchvision.utils.make_grid(inputs_i[0:8], padding=0),
            torchvision.utils.make_grid(adv_inputs_i[0:8], padding=0),
            torchvision.utils.make_grid(torch.clamp((adv_inputs_i[0:8] - inputs_i[0:8]) * 3 + 0.5, 0, 1), padding=0),
            torchvision.utils.make_grid(inputs_d[0:8], padding=0),
            torchvision.utils.make_grid(adv_inputs_d[0:8], padding=0),
            torchvision.utils.make_grid(torch.clamp((adv_inputs_d[0:8] - inputs_d[0:8]) * 3 + 0.5, 0, 1), padding=0)
        ], dim=1).detach())

        wandb.log({'images': [
                wandb.Image(torch.cat(successful_attacks, dim=2))]}, step=data_index )

    print('OVERALL')
    total_xi = torch.cat(xi)
    total_xd = torch.cat(xd)
    total_xip = torch.cat(xip)
    total_xdp = torch.cat(xdp)
    total_di = torch.cat(di)
    total_dd = torch.cat(dd)
    total_d = torch.cat(d)
    print("xi: %.2f xd: %.2f xi': %.2f xd': %.2f di: %.2f dd: %.2f d: %.2f" % (
    total_xi.mean().item(), total_xd.mean().item(), \
    total_xip.mean().item(), total_xdp.mean().item(), \
    total_di.mean().item(), total_dd.mean().item(), total_d.mean().item()))

if __name__ == '__main__':
    main()