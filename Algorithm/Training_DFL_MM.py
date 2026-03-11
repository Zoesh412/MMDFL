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
Global_Model_set = []
training_client = None
model_distribution = None
last_visit_round = None

def calculate_uniform_loss(a):
    if np.sum(a) == 0:
        return 0
    a = a / np.sum(a)
    uniform_vec = np.array([1 / len(a) for _ in range(len(a))])
    return np.linalg.norm(a - uniform_vec)

class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.args = args

    def calculate_label_distribuion(self, dataset):
        self.label_distribution = np.zeros(self.args.num_classes)
        ldr_train = DataLoader(DatasetSplit(dataset, self.data_idx), batch_size=self.args.local_bs, shuffle=True)
        for batch_idx, (images, labels) in enumerate(ldr_train):
            for label in labels:
                self.label_distribution[label] = self.label_distribution[label] + 1;

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
        return

def choose_next_neighbor(k, iter, args):
    global training_client, model_distribution, last_visit_round
    idx = training_client[k]

    if args.client_selection=='data_aware':
        model_dist = model_distribution[k]
        next_idx = -1
        min_uniform_loss = 1e9
        random_num = random.uniform(0, 1)
        if random_num >= args.curiosity:
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1:
                    next_model_dist = model_dist + Global_Client_set[nb_idx].label_distribution
                    uniform_loss = calculate_uniform_loss(next_model_dist)
                    if uniform_loss < min_uniform_loss:
                        next_idx = nb_idx
                        min_uniform_loss = uniform_loss
        else:
            nb_list = []
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1:
                    nb_list.append(nb_idx)
            next_idx = random.choice(nb_list)
        model_distribution[k] = model_distribution[k] + Global_Client_set[next_idx].label_distribution

    elif args.client_selection=="speed_aware":
        next_idx = -1
        min_epoch_time = 1e9
        random_num = random.uniform(0,1)
        if random_num >= args.curiosity:
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1:
                    epoch_time = get_communication_time(NetWork_type[idx][nb_idx]) + get_training_time(client_type_list[nb_idx])
                    if epoch_time < min_epoch_time:
                        next_idx = nb_idx
                        min_epoch_time = epoch_time
        else:
            nb_list = []
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1:
                    nb_list.append(nb_idx)
            next_idx = random.choice(nb_list)

    elif args.client_selection=="forget_aware":
        min_visit_epoch = 1e9
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                visit_epoch = last_visit_round[nb_idx]
                min_visit_epoch = min(min_visit_epoch, visit_epoch)
        nb_list = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                if last_visit_round[nb_idx] == min_visit_epoch:
                    nb_list.append(nb_idx)
        next_idx = random.choice(nb_list)

    elif args.client_selection == 'comprehensive':
        nb_list = []
        '''==========data_aware score=========='''
        model_dist = model_distribution[k]
        data_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                nb_list.append(nb_idx)
                next_model_dist = model_dist + Global_Client_set[nb_idx].label_distribution
                uniform_loss = calculate_uniform_loss(next_model_dist)
                data_aware_score.append(uniform_loss)
        sum_score = sum(data_aware_score)
        data_aware_score = [x / sum_score * 100 for x in data_aware_score]
        '''==========speed_aware score=========='''
        speed_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                epoch_time = get_communication_time(NetWork_type[idx][nb_idx]) + get_training_time(client_type_list[nb_idx])
                speed_aware_score.append(epoch_time)
        sum_score = sum(speed_aware_score)
        speed_aware_score = [x / sum_score * 100 for x in speed_aware_score]
        '''==========forget_aware score=========='''
        forget_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                visit_epoch = last_visit_round[nb_idx]
                forget_aware_score.append(visit_epoch)
        forget_aware_score = [(iter - x + 2)**(-0.5) for x in forget_aware_score]
        sum_score = sum(forget_aware_score)
        forget_aware_score = [x / sum_score * 100 for x in forget_aware_score]

        '''==========comprehensive score=========='''
        comprehensive_score = []
        # args.weight_data = max(0.7 * ((args.epochs - iter) / args.epochs), 0.2)
        # args.weight_speed = 0.1
        # args.weight_forget = 0.9 - args.weight_data
        for i in range(len(nb_list)):
            comprehensive_score.append(data_aware_score[i]*args.weight_data +
                                       speed_aware_score[i]*args.weight_speed +
                                       forget_aware_score[i]*args.weight_forget)

        next_idx = nb_list[comprehensive_score.index(min(comprehensive_score))]
        model_distribution[k] = model_distribution[k] + Global_Client_set[next_idx].label_distribution
    elif args.client_selection=='random':
        nb_list = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                nb_list.append(nb_idx)
        next_idx = random.choice(nb_list)

    return next_idx
def DFL_MM(args, net_glob, dataset_train, dataset_test, dict_users):
    global training_client, model_distribution, last_visit_round
    net_size = sum([param.nelement() for param in net_glob.parameters()])
    target_time = dict()
    target_comm = dict()
    target_acc1 = 55
    target_acc2 = 60
    acc = dict()
    model_cnt = int(args.num_users * args.frac)
    for idx in range(model_cnt):
        Global_Model_set.append(copy.deepcopy(net_glob))
        acc["model" + str(idx)] = []
    for idx in range(args.num_users):
        client = Client(idx, dict_users[idx], net_glob, args)
        client.calculate_label_distribuion(dataset_train)
        Global_Client_set.append(client)
    acc["acc"] = []
    time_consume = []
    comm_consume = []
    current_time = 0
    current_comm = 0
    training_client = random.sample(list(range(args.num_users)), model_cnt)
    model_distribution = [np.zeros(args.num_classes) for _ in range(model_cnt)]
    last_visit_round = [0 for _ in range(args.num_users)]
    for k, idx in enumerate(training_client):
        model_distribution[k] = model_distribution[k] + Global_Client_set[idx].label_distribution
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        if args.client_selection in ['data_aware', 'comprehensive']:
            print("+++++++model_distribution++++++")
            for k, m_d in enumerate(model_distribution):
                print(k,"th model distribution:", m_d)
        if args.client_selection in ['forget_aware', 'comprehensive']:
            print("+++++++visit_round+++++++")
            print(last_visit_round)
        print("choose client:", training_client)
        round_time = 0
        round_comm = 0

        if args.aggregation:
            agg_dict = dict()
            for k, idx in enumerate(training_client):
                if idx in agg_dict:
                    agg_dict[idx].append(copy.deepcopy(Global_Model_set[k]).state_dict())
                else:
                    agg_dict[idx] = []
                    agg_dict[idx].append(copy.deepcopy(Global_Model_set[k]).state_dict())

            for key in agg_dict.keys():
                if len(agg_dict[key]) > 1:
                    agg_model = Aggregation(agg_dict[key], [1 for _ in range(len(agg_dict[key]))])
                    for k, idx in enumerate(training_client):
                        if idx == key:
                            Global_Model_set[k].load_state_dict(agg_model)
                            model_distribution[k] = Global_Client_set[idx].label_distribution


        for k, (idx, model) in enumerate(zip(training_client, Global_Model_set)):
            Global_Client_set[idx].local_net = copy.deepcopy(model)
            local = LocalUpdate_DFL(args=args, dataset=dataset_train)
            local.train(client=Global_Client_set[idx], round=iter)
            Global_Model_set[k] = copy.deepcopy(Global_Client_set[idx].local_net)
            training_client[k] = choose_next_neighbor(k, iter, args)
            last_visit_round[training_client[k]] = iter+1
            round_time = max(round_time, get_training_time(client_type_list[idx]) + get_communication_time(NetWork_type[idx][training_client[k]]))
            round_comm = round_comm + (net_size * 8) / (1024 * 1024)  # MB
        current_time = current_time + round_time
        current_comm = current_comm + round_comm
        avg_acc = 0
        for idx in range(model_cnt):
            acc["model" + str(idx)].append(test(Global_Model_set[idx], dataset_test, args))
            avg_acc = avg_acc + acc["model" + str(idx)][-1]
            print("model ", idx, "acc: ", acc["model" + str(idx)][-1])
        acc["acc"].append(avg_acc / model_cnt)
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
    print(args.weight_data, " ", args.weight_speed, " ", args.weight_forget)


