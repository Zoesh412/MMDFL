#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
from torchvision import models
from vit_pytorch import SimpleViT

from Algorithm.Training_DFL import DFL
from Algorithm.Training_DFL_MM import DFL_MM
from Algorithm.Training_DFedPGP import DFedPGP
from Algorithm.Training_DFedSAM import DFedSAM
from Algorithm.Training_D_PSGD import D_PSGD
from Algorithm.Training_LD_SGD import LD_SGD
#from Algorithm.Training_OSGP import OSGP
from utils.options import args_parser
from utils.get_dataset import get_dataset
from utils.set_seed import set_random_seed
import torch
from models.SplitModel import *
matplotlib.use('Agg')

if __name__ == '__main__':
    # args initialize
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)

    # dataset initialize
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # model initialize
    client_net_list = []
    if 'resnet8' in args.model:
        net_glob = ResNet8_entire()
        share_net_glob = ResNet8_share()
        private_net_glob = ResNet8_private()
    if 'vgg' in args.model:
        net_glob = VGG16_entire()
    if 'mobilenet' in args.model:
        net_glob = mobilenet_entire()

    net_glob.apply(init_weights)
    share_net_glob.apply(init_weights)
    private_net_glob.apply(init_weights)

    if args.algorithm == 'DFL':
        DFL(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFL_MM':
        DFL_MM(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFedPGP':
        DFedPGP(args, share_net_glob, private_net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'D_PSGD':
        D_PSGD(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFedSAM':
        DFedSAM(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'LD_SGD':
        LD_SGD(args, net_glob, dataset_train, dataset_test, dict_users)
    else:
        raise "%s algorithm has not achieved".format(args.algorithm)
