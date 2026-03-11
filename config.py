import os
import pickle
import random

import numpy as np
from utils.options import args_parser


args = args_parser()
"+++++++++++++++++++++client config+++++++++++++++++++++++++"
# client_type_list = ["weak"]*10 + ["middle"]*6 + [ ]
# for i in range(args.num_users):
#     client_type = random.choice(['weak', 'middle', 'strong'])
#     client_type_list.append(client_type)
# print(client_type_list)
client_type_list = ['strong', 'strong', 'weak', 'strong', 'middle', 'weak', 'middle', 'strong', 'strong', 'weak', 'weak',
                        'middle', 'weak', 'weak', 'weak', 'strong', 'middle', 'strong', 'strong', 'middle']

#client_type_list = ['strong', 'strong', 'middle', 'strong', 'weak', 'middle', 'middle', 'weak', 'weak', 'weak', 'strong', 'middle', 'middle', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'strong', 'middle', 'weak', 'weak', 'middle', 'weak', 'weak', 'middle', 'strong', 'weak', 'weak', 'middle', 'strong', 'middle', 'middle', 'weak', 'weak', 'strong', 'strong', 'strong', 'middle']
#client_type_list = ['middle', 'strong', 'strong', 'middle', 'middle', 'strong', 'strong', 'weak', 'strong', 'middle', 'middle', 'middle', 'weak', 'middle', 'strong', 'weak', 'weak', 'weak', 'strong', 'weak', 'middle', 'weak', 'strong', 'weak', 'middle', 'weak', 'weak', 'middle', 'weak', 'middle', 'middle', 'middle', 'strong', 'strong', 'strong', 'weak', 'strong', 'strong', 'weak', 'weak', 'middle', 'weak', 'middle', 'strong', 'middle', 'strong', 'middle', 'middle', 'weak', 'weak', 'middle', 'middle', 'middle', 'strong', 'middle', 'weak', 'weak', 'weak', 'middle', 'strong', 'strong', 'weak', 'middle', 'strong', 'strong', 'strong', 'weak', 'middle', 'weak', 'strong', 'strong', 'weak', 'weak', 'strong', 'weak', 'middle', 'weak', 'weak', 'middle', 'middle']
def get_training_time(client_type):
    if client_type == 'weak':
        client_training_time = np.random.normal(50,2)
    if client_type == 'middle':
        client_training_time = np.random.normal(20,2)
    if client_type == 'strong':
        client_training_time = np.random.normal(10,2)
    return client_training_time
"+++++++++++++++++++++++topo config+++++++++++++++++++++++++"
Adjacency_matrix = np.zeros((args.num_users, args.num_users))
if args.topo == 'ring':
    for idx in range(args.num_users):
        left = (idx - 1 + args.num_users) % args.num_users
        right = (idx + 1) % args.num_users
        Adjacency_matrix[idx][left] = 1
        Adjacency_matrix[idx][right] = 1

if args.topo == 'M-ring':
    if os.path.exists('./topo/M-ring-{}.pkl'.format(args.num_users)):
        print("loaded exist topo")
        with open('./topo/M-ring-{}.pkl'.format(args.num_users), "rb") as file:
            Adjacency_matrix = pickle.load(file)
    else:
        print("generate a new topo")
        for idx in range(args.num_users):
            left = (idx - 1 + args.num_users) % args.num_users
            right = (idx + 1) % args.num_users
            Adjacency_matrix[idx][left] = 1
            Adjacency_matrix[idx][right] = 1
        for idx in range(args.num_users):
            new_edge_cnt = random.randint(0,2)
            left = (idx - 1 + args.num_users) % args.num_users
            right = (idx + 1) % args.num_users
            nb_list = list(range(args.num_users))
            nb_list.remove(idx)
            nb_list.remove(left)
            nb_list.remove(right)
            neighbor_client = random.sample(nb_list, new_edge_cnt)
            for nb_idx in neighbor_client:
                Adjacency_matrix[idx][nb_idx] = 1
                Adjacency_matrix[nb_idx][idx] = 1
        with open('./topo/M-ring-{}.pkl'.format(args.num_users), "wb") as file:
            pickle.dump(Adjacency_matrix, file)

"+++++++++++++++++++++++bandwith config+++++++++++++++++++++++++"
if os.path.exists('./topo/{}-{}-network.pkl'.format(args.topo, args.num_users)):
    print("loaded exist network")
    with open('./topo/{}-{}-network.pkl'.format(args.topo, args.num_users), "rb") as file:
        NetWork_type = pickle.load(file)
else:
    print("generate a new network_type")
    NetWork_type = [["" for _ in range(args.num_users)] for _ in range(args.num_users)]
    for idx in range(args.num_users):
        for jdx in range(idx+1, args.num_users):
            if Adjacency_matrix[idx][jdx] == 1:
                client_type = np.random.choice(['weak', 'middle', 'strong'], p=[0.2,0.3,0.5])
                NetWork_type[idx][jdx] = client_type
                NetWork_type[jdx][idx] = client_type
    with open('./topo/{}-{}-network.pkl'.format(args.topo, args.num_users), "wb") as file:
        pickle.dump(NetWork_type, file)

def get_communication_time(net_type):
    if net_type == 'weak':
        communication_time = np.random.normal(10, 0.5)
    if net_type == 'middle':
        communication_time = np.random.normal(5, 0.5)
    if net_type == 'strong':
        communication_time = np.random.normal(2, 0.5)
    return communication_time


