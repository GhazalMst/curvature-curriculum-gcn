import variables
import pickle
import torch
import numpy as np
import random
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import pandas as pd
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx) 
    return mx


def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def prepare_dataset(triples, node_features, edge_index, ir_scores=None):
    data_list = []
    
    # 1. Konvertiere edge_index (train_positive) korrekt für PyTorch Geometric
    # Wir nehmen nur die ersten beiden Spalten [:, :2], transponieren zu [2, Kanten]
    # und machen einen torch.long Tensor daraus.
    if isinstance(edge_index, np.ndarray):
        edge_index_tensor = torch.tensor(edge_index[:, :2], dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = edge_index.to(torch.long)

    for i in range(len(triples)):
        u, v, label = triples[i]

        score = ir_scores[i] if ir_scores is not None else torch.zeros(768)

        # Falls u und v noch keine Tensoren oder Strings sind, als Standard-Typ sichern
        # data.key erwartet Strings oder Listen von Strings in deiner train.py
        edge_key = [str(int(u)), str(int(v))]

        d = Data(
            x=node_features,                               # Knotenfeatures [Anzahl_Knoten, Feature_Dim]
            edge_index=edge_index_tensor,                  # JETZT REIN: [2, Anzahl_Kanten] als torch.long
            y=torch.tensor([label], dtype=torch.float32),  # Label als Float-Tensor
            set_indices=torch.tensor([[u, v]], dtype=torch.long), 
            ir_score=score.unsqueeze(0),
            key=edge_key                                   # Wichtig für seperate_seen_from_new_edges in train.py!
        )
        data_list.append(d)
    return data_list

def load_datasets(args):
    dataset = args.dataset
    
    if args.dataset == "huri":
        positive = np.loadtxt('../data/HuRI2.edgelist', dtype=np.int64)
        G_all = nx.Graph()
        G_all.add_edges_from(positive)
        if positive[0].shape==1:
            unique_entity = positive[0]
        else:
            unique_entity = len(G_all.nodes)
        negative_all = list(nx.non_edges(G_all))
        np.random.seed(args.seed)
        np.random.shuffle(negative_all)
        negative = np.asarray(negative_all[:positive.shape[0]])
        val_ratio, test_ratio = 0.1, 0.2
        val_size = int(val_ratio * positive.shape[0])
        test_size = int(test_ratio * positive.shape[0])
        positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)
        negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)
        
        train_positive = positive[: -(val_size + test_size)]
        val_positive = positive[-(val_size + test_size) : -test_size]
        test_positive = positive[: -test_size]
        
    
        train_data = np.vstack((positive[: -(val_size + test_size)], negative[: -(val_size + test_size)]))
        val_data = np.vstack((positive[-(val_size + test_size): -test_size], negative[-(val_size + test_size): -test_size]))
        test_data = np.vstack((positive[-test_size:], negative[-test_size:]))        
        
        adj = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                        shape=(unique_entity, unique_entity), dtype=np.float32)
        adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = adj_o.todense()
        features_o = normalize(features)
        x_o = torch.tensor(features_o, dtype=torch.float)
        
        train_set = prepare_dataset(train_data, x_o, train_positive)
        val_set = prepare_dataset(val_data, x_o, val_positive)
        test_set = prepare_dataset(test_data, x_o, test_positive)

    if args.dataset == "pgr":
        dataloader_loc = variables.dir_data + "/{}_data/{}_train_test_val_doc2vec_v2.pkl".format(dataset, dataset)
        train_set, test_set, val_set = pickle.load(open(dataloader_loc, "rb"))
    elif args.dataset == "gdpr":
        
        dataloader_loc = variables.dir_data + "/omim_data/omim_train_test_val_doc2vec.pkl"
        train_set, val_set, test_set = pickle.load(open(dataloader_loc, "rb"))

    print("Length of Train size = {}, val size = {}, test size = {}".format(len(train_set), len(val_set), len(test_set)))

    return train_set, val_set, test_set


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model_name(args):
    dataset = args.dataset
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    curriculum_length = args.curriculum_length
    seed = args.seed
    approach = args.prioritizing_approach
    lc = args.loss_creteria
    mo = args.metric_order
    km = args.use_k_means
    random = args.add_random
    eval_test = args.evaluate_test_per_epoch
    model_var_order = [
        dataset,
        lr,
        L2,
        add_additional_feature,
        curriculum_length,
        seed,
        approach,
        lc,
        mo,
        km,
        random,
        eval_test
    ]

    model_name = "{}_{}_{}_addF_{}_cl_{}_s_{}_app_{}_{}_{}_km_{}_r_{}_eval_{}".format(
        *model_var_order)

    return model_name


def save_the_best_model(model, epoch, optimizer, performance, args):
    model_name = get_model_name(args)
    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  "performance": performance}

    torch.save(checkpoint, variables.dir_model + '/{}_best.pth'.format(model_name))

    print("model saved.")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")