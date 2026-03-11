
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

Global_Client_set = []

class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.trans_net = copy.deepcopy(net)


class LocalUpdate_DFL(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
    def train(self, client, round):
        net = copy.deepcopy(client.local_net)
        net.train()
        net = net.to(self.args.device)
        ldr_train = DataLoader(DatasetSplit(self.dataset, client.data_idx), batch_size=self.args.local_bs, shuffle=True)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
        net = net.to('cpu')
        client.local_net = copy.deepcopy(net)
        client.trans_net = copy.deepcopy(net)
        return


def DFL(args, net_glob, dataset_train, dataset_test, dict_users):
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
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        # training of each client
        for idx in range(args.num_users):
            local = LocalUpdate_DFL(args=args, dataset=dataset_train)
            local.train(client=Global_Client_set[idx],round=iter)
        # communicate with neighboring clients to update local model
        avg_acc = 0
        round_time = 0
        round_comm = 0
        for idx in range(args.num_users):
            w_locals = []
            lens = []
            t = get_training_time(client_type_list[idx])
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1:
                    w_locals.append(Global_Client_set[nb_idx].trans_net.state_dict())
                    lens.append(Global_Client_set[nb_idx].data_cnt)
                    t = t + get_communication_time(NetWork_type[idx][nb_idx])
                    round_comm = round_comm + (net_size*8)/(1024*1024) #MB
            round_time = max(round_time, t)
            w_locals.append(Global_Client_set[idx].trans_net.state_dict())
            lens.append(Global_Client_set[idx].data_cnt)

            w_agg = Aggregation(w_locals, lens)
            Global_Client_set[idx].local_net.load_state_dict(w_agg)
            acc["client" + str(idx)].append(test(Global_Client_set[idx].local_net, dataset_test, args))
            avg_acc = avg_acc + acc["client" + str(idx)][-1]
            print("client ", idx, "acc: ", acc["client" + str(idx)][-1])
        current_time = current_time + round_time
        current_comm = current_comm + round_comm
        acc["acc"].append(avg_acc/args.num_users)
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





