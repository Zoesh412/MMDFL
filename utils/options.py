#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--personal_ep', type=int, default=1, help="the number of personal model epochs: E")
    parser.add_argument('--shared_ep', type=int, default=5, help="the number of shared model epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=50, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.998, help="learning rate decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay (default: 1e-4)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="DFL_MM")
    parser.add_argument("--cifar100_coarse", type=int, default=0, help="use 20 class cifar100")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet8', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=0, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=5, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=1.0,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_sizes', type=int, default=32, help="number of sizes of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # topo structure
    parser.add_argument('--topo', type=str, default='M-ring', help="topo structure of communication")
    parser.add_argument('--client_selection', type=str, default='comprehensive', help='strategy of client selection')
    parser.add_argument('--weight_data', type=float, default='0.8', help='weight of data aware')
    parser.add_argument('--weight_speed', type=float, default='0.0', help='weight of speed aware')
    parser.add_argument('--weight_forget', type=float, default='0.2', help='weight of forget aware')
    parser.add_argument('--curiosity', type=float, default=0.2, help='The probability of violating the client selection strategy')
    parser.add_argument('--aggregation', type=bool, default=True, help='whether aggregation when different model meeting')
    args = parser.parse_args()
    return args
