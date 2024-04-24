import numpy as np
import torch
import torch.utils.data as Data
import networkx as nx
import pickle
import scipy.sparse as sp

dataname = "weibo"

file_path = 'dataset/' + dataname + '/'

def u2idx(nodes):
    return {node:i+1 for i,node in enumerate(nodes)}


def obtain_user_dict(file_path):
    try:
        with open(file_path+"u2idx.pkl", 'rb') as f:
            u2idx_dict = pickle.load(f)
    except:
        train_file = file_path + "cascade_train1.txt"
        val_file = file_path + "cascade_validation1.txt"
        test_file = file_path + "cascade_test1.txt"
        nodes = set()
        with open(train_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        with open(val_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        with open(test_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        nodes = list(nodes)
        u2idx_dict = u2idx(nodes)
        with open(file_path+"u2idx.pkl", 'wb') as f:
            pickle.dump(u2idx_dict, f)

    return u2idx_dict
'''
def obtain_user_dict(file_path):
    train_file = file_path + "cascade_train.txt"
    val_file = file_path + "cascade_validation.txt"
    test_file = file_path + "cascade_test.txt"
    nodes = set()
    with open(train_file, 'r') as casf:
        for line in casf:
            parts = line.strip().split('\t')
            edges = parts[-2].split(" ")
            for edge in edges:
                edge = edge.split(":")
                nodes.add(edge[0])
                nodes.add(edge[1])
    with open(val_file, 'r') as casf:
        for line in casf:
            parts = line.strip().split('\t')
            edges = parts[-2].split(" ")
            for edge in edges:
                edge = edge.split(":")
                nodes.add(edge[0])
                nodes.add(edge[1])
    with open(test_file, 'r') as casf:
        for line in casf:
            parts = line.strip().split('\t')
            edges = parts[-2].split(" ")
            for edge in edges:
                edge = edge.split(":")
                nodes.add(edge[0])
                nodes.add(edge[1])
    nodes = list(nodes)
    u2idx_dict = u2idx(nodes)
    with open(file_path+"u2idx.pkl", 'wb') as f:
        pickle.dump(u2idx_dict, f)

    return u2idx_dict
'''
def obtain_N():
    u2idx_dict = obtain_user_dict(file_path)
    return len(u2idx_dict)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def load_cascade(file_path, file_type=0):
    # 0 train, 1 val, 2 test
    if file_type == 0:
        file_name = file_path + "cascade_shortestpath_train1.txt"
    elif file_type == 1:
        file_name = file_path + "cascade_shortestpath_validation1.txt"
    else:
        file_name = file_path + "cascade_shortestpath_test1.txt"
    user_dict = obtain_user_dict(file_path)

    cascades_info = {}

    with open(file_name, 'r') as casf:
        for line in casf:
            cascade_lst = []
            timespan_lst = []
            g = nx.DiGraph()
            # cascade id，origin node, edges, labels
            parts = line.strip().split('\t')
            # add label
            label = np.log(int(parts[-1])+1)/np.log(2.0)
            cascade_id = parts[0]
            # construct cascade graph
            edges = parts[1:-1]
            for i, edge in enumerate(edges):
                edge = edge.split(":")
                if i == 0:
                    cascade_lst.append(user_dict[edge[0]])
                    timespan_lst.append(float(edge[-1]))
                    continue
                time = float(edge[-1])
                edge = edge[0].split(',')

                cascade_lst.append(user_dict[edge[-1]])
                timespan_lst.append(time)
                for i in range(len(edge) - 1):
                    g.add_edge(user_dict[edge[i]], user_dict[edge[i+1]])

            adj = nx.adjacency_matrix(g)
            adj = adj + sp.eye(adj.shape[0])
            adj = adj.todense()
            assert len(cascade_lst) == len(timespan_lst)
            if len(adj) > len(cascade_lst):
                continue
            cascades_info[cascade_id] = [g, cascade_lst, timespan_lst, adj, label]
    return cascades_info


class CascadeData(Data.Dataset):
    def __init__(self, cas_src, time_src, adj_list, tgt):
        super(CascadeData, self).__init__()
        self.cas_src = cas_src
        self.time_src = time_src
        self.adj_list = adj_list
        self.tgt = tgt

    def __getitem__(self, item):
        return self.cas_src[item], self.time_src[item], self.adj_list[item], self.tgt[item]

    def __len__(self):
        return len(self.cas_src)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    cas_src, time_src, adj_list, tgt = zip(*batch)
    pad_src = []
    pad_time = []
    pad_adj_list = []
    lens = []
    real_len = [len(s) for s in cas_src]
    max_len = max(real_len)
    for s in cas_src:
        temp_src = [0] * max_len
        temp_src[:len(s)] = s
        pad_src.append(temp_src)
        lens.append(len(s))

    for t in time_src:
        temp_tsrc = [0] * max_len
        temp_tsrc[:len(t)] = t
        pad_time.append(temp_tsrc)

    for adj in adj_list:
        pad_adj = np.zeros((max_len, max_len))
        n = len(adj)
        pad_adj[:n,:n] = adj
        pad_adj_list.append(pad_adj)


    return torch.LongTensor(pad_src), torch.FloatTensor(pad_time), torch.FloatTensor(pad_adj_list), torch.FloatTensor(tgt), torch.LongTensor(real_len)

def creat_dataloader(file_type, batch_size, shuffle=True):
    cascades_info = load_cascade(file_path, file_type)
    # idx = []
    cas_src = []
    time_src = []
    tgt = []
    adj_list = []
    #  [g, cascade_lst, timespan_lst, adj, label]
    for k, v in cascades_info.items():
        cas_src.append(v[1])
        time_src.append(v[2])
        adj_list.append(v[3])
        tgt.append(v[-1])
    dataset = CascadeData(cas_src, time_src, adj_list, tgt)
    loader = Data.DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader
