#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os
import pickle


def save_result(data, ylabel, args):
    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        if args.client_selection != 'comprehensive':
            file = '{}_{}_{}_{}_{}_lr{}_{}_numuser{}_clientSelection_{}'.format(ylabel, args.algorithm, args.dataset, args.model, args.epochs,
                                                        args.lr, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), args.num_users, args.client_selection)
        else:
            file = '{}_{}_{}_{}_{}_lr{}_{}_numuser{}_clientSelection_{}_{}_{}_{}'.format(ylabel, args.algorithm, args.dataset,
                                                                                args.model, args.epochs,
                                                                                args.lr,
                                                                                datetime.datetime.now().strftime(
                                                                                    "%Y_%m_%d_%H_%M_%S"),
                                                                                args.num_users, args.client_selection,
                                                                                args.weight_data, args.weight_speed, args.weight_forget)
    else:
        path += '/{}'.format(args.data_beta)
        if args.client_selection != 'comprehensive':
            file = '{}_{}_{}_{}_{}_lr{}_{}_numuser{}_clientSelection_{}'.format(ylabel, args.algorithm, args.dataset,
                                                                                args.model, args.epochs,
                                                                                args.lr,
                                                                                datetime.datetime.now().strftime(
                                                                                    "%Y_%m_%d_%H_%M_%S"),
                                                                                args.num_users, args.client_selection)
        else:
            file = '{}_{}_{}_{}_{}_lr{}_{}_numuser{}_clientSelection_{}_{}_{}_{}'.format(ylabel, args.algorithm,
                                                                                         args.dataset,
                                                                                         args.model, args.epochs,
                                                                                         args.lr,
                                                                                         datetime.datetime.now().strftime(
                                                                                             "%Y_%m_%d_%H_%M_%S"),
                                                                                         args.num_users,
                                                                                         args.client_selection,
                                                                                         args.weight_data,
                                                                                         args.weight_speed,
                                                                                         args.weight_forget)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path,file), 'wb') as file:
        pickle.dump(data, file)
    print('save finished')
    file.close()
