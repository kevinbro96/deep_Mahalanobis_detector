"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
import pdb

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | PGD | CW')
parser.add_argument('--vae_path', default='./data/96.32/model_epoch252.pth', help='folder to output results')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    pre_trained_net = './Pretrain_Subspace/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        
    # load networks
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
    elif args.net_type == 'resnet':
        # model = models.ResNet34(num_c=args.num_classes)
        # model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])

    model = models.Wide_ResNet(28, 10, 0.3, 10)
    model = nn.DataParallel(model)
    model_dict = model.state_dict()
    save_model = torch.load(args.vae_path)
    state_dict = {k.replace('classifier.',''): v for k, v in save_model.items() if k.replace('classifier.','') in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()
    print('load model: ' + args.net_type)
    vae = models.CVAE(d=32, z=2048)
    vae = nn.DataParallel(vae)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)
    vae.cuda()
    vae.eval()

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    test_clean_data = torch.load(args.outf + 'clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_adv_data = torch.load(args.outf + 'adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_noisy_data = torch.load(args.outf + 'noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_label = torch.load(args.outf + 'label_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.module.feature_list(temp_x-vae(temp_x))[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, vae, args.num_classes, feature_list, train_loader)

    print('get LID scores')
    LID, LID_adv, LID_noisy \
    = lib_generation.get_LID(model, vae, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output)
    overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    list_counter = 0
    for overlap in overlap_list:
        Save_LID = np.asarray(LID[list_counter], dtype=np.float32)
        Save_LID_adv = np.asarray(LID_adv[list_counter], dtype=np.float32)
        Save_LID_noisy = np.asarray(LID_noisy[list_counter], dtype=np.float32)
        Save_LID_pos = np.concatenate((Save_LID, Save_LID_noisy))
        LID_data, LID_labels = lib_generation.merge_and_generate_labels(Save_LID_adv, Save_LID_pos)
        file_name = os.path.join(args.outf, 'LID_%s_%s_%s.npy' % (overlap, args.dataset, args.adv_type))
        LID_data = np.concatenate((LID_data, LID_labels), axis=1)
        np.save(file_name, LID_data)
        list_counter += 1

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('\nNoise: ' + str(magnitude))
        for i in range(num_output):
            M_in \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_clean_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_adv_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_noisy \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_noisy_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_noisy = np.asarray(M_noisy, dtype=np.float32)
            if i == 0:
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            else:
                Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
        file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset, args.adv_type))

        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)

    print('get KD scores')
    band_list = [100, 500, 1000]
    train_example = lib_generation.KD_extracter(model, vae, args.num_classes, feature_list, train_loader) # 60000 x 2048
    for band in band_list:
        print('\nBand: ' + str(band))
        K_in \
            = lib_generation.get_KD(model, vae, test_clean_data, test_label, \
                                                       train_example, -1, band)
        K_in = np.asarray(K_in, dtype=np.float32)
        KernelDesity_in = K_in.reshape((K_in.shape[0], -1))


        K_out \
            = lib_generation.get_KD(model, vae, test_adv_data, test_label, \
                                                       train_example, -1, band)
        K_out = np.asarray(K_out, dtype=np.float32)
        KernelDesity_out = K_out.reshape((K_out.shape[0], -1))


        K_noisy \
            = lib_generation.get_KD(model, vae, test_noisy_data, test_label, \
                                                       train_example, -1, band)
        K_noisy = np.asarray(K_noisy, dtype=np.float32)
        KernelDesity_noisy = K_noisy.reshape((K_noisy.shape[0], -1))

        KernelDesity_in = np.asarray(KernelDesity_in, dtype=np.float32)
        KernelDesity_out = np.asarray(KernelDesity_out, dtype=np.float32)
        KernelDesity_noisy = np.asarray(KernelDesity_noisy, dtype=np.float32)
        KernelDesity_pos = np.concatenate((KernelDesity_in, KernelDesity_noisy))

        KernelDesity_data, KernelDesity_labels = lib_generation.merge_and_generate_labels(KernelDesity_out,
                                                                                        KernelDesity_pos)
        file_name = os.path.join(args.outf, 'KernelDensity_%s_%s_%s.npy' % (str(band), args.dataset, args.adv_type))

        KernelDesity_data = np.concatenate((KernelDesity_data, KernelDesity_labels), axis=1)
        np.save(file_name, KernelDesity_data)

if __name__ == '__main__':
    main()

