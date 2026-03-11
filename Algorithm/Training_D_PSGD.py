#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import datetime
import pickle
import random

import matplotlib
from thop import profile
matplotlib.use('Agg')
from utils.utils import save_result
from config import *
import torch
from torch import nn, autograd
import numpy as np
from utils.FL_utils import *
from utils.FL_utils import DataLoader
from tqdm import tqdm

Global_Client_set = []

class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.trans_net = copy.deepcopy(net)

def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)

def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

class LocalUpdate_D_PSGD(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
    def train(self, round, net_size):
        # statistics of communication time
        times = dict()
        round_comm = 0
        optimizer_dict = dict()
        for idx in range(args.num_users):
            times[idx] = 0
            if self.args.optimizer == 'sgd':
                optimizer_dict[idx] = torch.optim.SGD(Global_Client_set[idx].local_net.parameters(),
                                                      lr=self.args.lr * (self.args.lr_decay ** round),
                                                      momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        for iter in range(self.args.local_ep):
            max_batch_idx = 0
            ldr_train_dict = dict()
            for idx in range(args.num_users):
                ldr_train_dict[idx] = DataLoader(DatasetSplit(self.dataset,Global_Client_set[idx].data_idx), batch_size=self.args.local_bs, shuffle=True)
                max_batch_idx = max(max_batch_idx, len(ldr_train_dict[idx]))
            for current_batch_idx in range(max_batch_idx):
                #print("current_batch_idx:", iter,"/", current_batch_idx)
                # all client calculate the gradients
                #print("+++all client calculate the gradients+++")
                for idx in range(args.num_users):
                    Global_Client_set[idx].local_net.train()
                    Global_Client_set[idx].local_net.to(self.args.device)
                    Global_Client_set[idx].local_net.zero_grad()
                    for batch_idx, (images, labels) in enumerate(ldr_train_dict[idx]):
                        if batch_idx == current_batch_idx:
                            images, labels = images.to(self.args.device), labels.to(self.args.device)
                            Global_Client_set[idx].local_net.zero_grad()
                            log_probs = Global_Client_set[idx].local_net(images)['output']
                            loss = self.loss_func(log_probs, labels)
                            loss.backward()
                # Then, processing the model aggregation
                #print("+++all client processing the model aggregation+++")
                for idx in range(args.num_users):
                    net_flat = flatten(Global_Client_set[idx].local_net)
                    cnt = 0
                    for nb_idx in range(args.num_users):
                        if Adjacency_matrix[idx][nb_idx] == 1:
                            Global_Client_set[nb_idx].trans_net.to(self.args.device)
                            net_flat.add_(flatten(Global_Client_set[nb_idx].trans_net))
                            times[idx] = times[idx] + get_communication_time(NetWork_type[idx][nb_idx])
                            if iter == 0:
                                round_comm = round_comm + (net_size * 8) / (1024 * 1024)  # MB
                            cnt = cnt + 1
                    net_flat.div_((cnt+1))
                    unflatten(Global_Client_set[idx].local_net, net_flat)
                # Last, processing the model update
                #print("+++all client processing the model update+++")
                for idx in range(args.num_users):
                    optimizer_dict[idx].step()
                    Global_Client_set[nb_idx].local_net.to('cpu')
                    Global_Client_set[idx].trans_net = copy.deepcopy(Global_Client_set[idx].local_net)

        return times, round_comm


def D_PSGD(args, net_glob, dataset_train, dataset_test, dict_users):
    net_size = sum([param.nelement() for param in net_glob.parameters()])
    target_time = dict()
    target_comm = dict()
    target_acc1 = 55
    target_acc2 = 60
    acc = dict()
    for idx in range(args.num_users):
        client = Client(idx, dict_users[idx], net_glob, args)
        Global_Client_set.append(client)
        acc["client" + str(idx)] = []
    acc["acc"] = []
    time_consume = []
    comm_consume = []
    current_time = 0
    current_comm = 0
    for iter in tqdm(range(args.epochs)):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        local = LocalUpdate_D_PSGD(args=args, dataset=dataset_train)
        times, round_comm = local.train(round=iter, net_size=net_size)
        avg_acc = 0
        round_time = 0
        for idx in range(args.num_users):
            t = get_training_time(client_type_list[idx])
            #round_time = max(round_time, max(t, times[idx]))
            round_time = max(round_time, t)
            acc["client" + str(idx)].append(test(Global_Client_set[idx].local_net, dataset_test, args))
            avg_acc = avg_acc + acc["client" + str(idx)][-1]
            print("client ", idx, "acc: ", acc["client" + str(idx)][-1])
        current_time = current_time + round_time
        current_comm = current_comm + round_comm
        acc["acc"].append(avg_acc / args.num_users)
        time_consume.append(current_time)
        comm_consume.append(current_comm)
        print("acc acc: ", acc["acc"][-1])
        if acc["acc"][-1] >= target_acc1:
            if target_acc1 not in target_time:
                target_time[target_acc1] = time_consume[-1]
            if target_acc1 not in target_comm:
                target_comm[target_acc1] = comm_consume[-1]
        if acc["acc"][-1] >= target_acc2:
            if target_acc2 not in target_time:
                target_time[target_acc2] = time_consume[-1]
            if target_acc2 not in target_comm:
                target_comm[target_acc2] = comm_consume[-1]
    save_result(acc, 'test_acc', args)
    save_result(time_consume, 'time', args)
    save_result(comm_consume, 'comm', args)
    print("target_time:", target_time)
    print("target_comm:", target_comm)
    for key in acc.keys():
        print(key)
        avg_acc_and_var(acc[key])
