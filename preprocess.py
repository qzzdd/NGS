import gc
import numpy as np
import torch
import pickle as pkl
import scipy.sparse as sp
from scipy.io import loadmat

"""
    Read and process data
"""


cstr_nc = {
    "amazon": [0, 1, 2, 3],
    "yelp": [0, 1, 2, 3],
    #"amazon": [0, 1, 2, 3, 4],  # including homo adj
    #"yelp": [0, 1, 2, 3, 4],  # including homo adj
    "mimic": [0, 1, 2, 3, 4]
}


def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dict_list_to_sparse_adj(relation_dict):
    rows = []
    cols = []
    for source in relation_dict:
        rows += [source] * len(list(relation_dict[source]))
        cols += list(relation_dict[source])
    values = [1] * len(rows)
    return sp.csr_matrix((values, (rows, cols)), dtype=np.float32)


def load_data(prefix, data):
    assert (data in ['amazon', 'yelp', 'mimic'])
    # * load data
    if data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        node_feats = data_file['features'].todense().A
        # load the adj_lists
        edges = [data_file['net_upu'], data_file['net_usu'], data_file['net_uvu']]

    elif data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        node_feats = data_file['features'].todense().A
        # load the adj_lists
        edges = [data_file['net_rur'], data_file['net_rtr'], data_file['net_rsr']]

    elif data == 'mimic':
        data_file = loadmat(prefix + 'Mimic.mat')
        labels = data_file['label'].flatten()
        # load the preprocessed adj_lists
        adjs_pt = []
        with open(prefix + 'mic_vmv_adjlists.pickle', 'rb') as file:
            relation1 = pkl.load(file)
            mx = dict_list_to_sparse_adj(relation1)
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx)).cuda())
            del relation1
            gc.collect()
        with open(prefix + 'mic_vav_adjlists.pickle', 'rb') as file:
            relation2 = pkl.load(file)
            mx = dict_list_to_sparse_adj(relation2)
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx)).cuda())
            del relation2
            gc.collect()
        with open(prefix + 'mic_vpv_adjlists.pickle', 'rb') as file:
            relation3 = pkl.load(file)
            mx = dict_list_to_sparse_adj(relation3)
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx)).cuda())
            del relation3
            gc.collect()
        with open(prefix + 'mic_vdv_adjlists.pickle', 'rb') as file:
            relation4 = pkl.load(file)
            mx = dict_list_to_sparse_adj(relation4)
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx)).cuda())
            del relation4
            gc.collect()
        # edges = [relation1, relation2, relation3, relation4]

    if data in ['amazon', 'yelp']:
        adjs_pt = []
        for mx in edges:
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(
                mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_pt[0].shape[0], dtype=np.float32).tocoo()).cuda())
        adjs_pt.append(torch.sparse.FloatTensor(size=adjs_pt[0].shape).cuda())
        node_feats = torch.from_numpy(node_feats.astype(np.float32)).cuda()
    else:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_pt[0].shape[0], dtype=np.float32).tocoo()).cuda())
        adjs_pt.append(torch.sparse.FloatTensor(size=adjs_pt[0].shape).cuda())
        node_feats = torch.from_numpy(node_feats.astype(np.float32)).cuda()
    print("Loading {} adjs...".format(len(adjs_pt)))

    return adjs_pt, node_feats, labels


def numpy_to_torch(train_idx, valid_idx, test_idx, train_target, valid_target, test_target):
    train_idx = torch.from_numpy(train_idx).type(torch.long)
    valid_idx = torch.from_numpy(valid_idx).type(torch.long)
    test_idx = torch.from_numpy(test_idx).type(torch.long)

    train_target = torch.from_numpy(train_target).type(torch.long).cuda()
    valid_target = torch.from_numpy(valid_target).type(torch.long).cuda()
    test_target = torch.from_numpy(test_target).type(torch.long).cuda()

    return train_idx, valid_idx, test_idx, train_target, valid_target, test_target


def normalize(mx):
    """
        Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
