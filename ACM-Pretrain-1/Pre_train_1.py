import numpy as np
import argparse
import os.path as osp
import random
import nni
from itertools import chain
import scipy.io as sio

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from sklearn.cluster import KMeans
from evaluation.metrics import clustering_metrics


def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data_edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data_edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data_x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data_x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data_x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data_x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    GL = np.squeeze(data_y)
    z = model(data_x, data_edge_index)
    # print(type(data.edge_index), data.edge_index.shape)
    FX = np.squeeze(z.cpu().detach().numpy())
    # print(type(FX), FX.shape)
    nb_classes = 3
    Pred = KMeans(n_clusters=nb_classes, random_state=9).fit_predict(FX) + 1
    cm = clustering_metrics(GL, Pred)
    acc, f1_macro, precision_macro, nmi, adjscore, _ = cm.evaluationClusterModelFromLabel()
    return acc, f1_macro, nmi, adjscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='ACM')
    parser.add_argument('--param', type=str, default='local:FACM.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--save_name', type=str, default='Pre_train_1.pkl', help='save ckpt name')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    dataset = sio.loadmat("ACM.mat")
    data_x = torch.tensor(dataset['X1'], dtype=torch.float)
    data_y = dataset['Y']
    data_edge_index = dataset['A1']
    A_dict = {i: np.nonzero(row)[0].tolist() for i, row in enumerate(data_edge_index)}
    links = list(A_dict.values())
    edges = [[(i, j) for j in js] for i, js in enumerate(links)]
    edges = list(chain(*edges))
    data_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data_num_nodes = 3025
    data_num_features = 1870
    data_x = data_x.to("cuda")
    data_edge_index = data_edge_index.to("cuda")

    # generate split
    split = generate_split(data_num_nodes, train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    encoder = Encoder(data_num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data_edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data_edge_index, aggr='sink', k=200).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data_edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data_x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data_x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data_edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data_x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data_x, node_c=node_pr).to(device)

    log = args.verbose.split(',')

    best = 0

    for epoch in range(1, param['num_epochs'] + 1):
        loss = train()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
            if epoch % 100 == 0:
                acc, f1_macro, nmi, adjscore = test()
                if acc > best:
                    best = acc
                    best_t = epoch
                    torch.save(model.state_dict(), args.save_name)
                if 'eval' in log:
                    print('-' * 100)
                    print("Rearrange:", "\033[1;31;43m Acc:%.4f \033[0m" % acc)
                    print("Rearrange:", "\033[1;31;43m Fsc:%.4f \033[0m" % f1_macro)
                    print("Rearrange:", "\033[1;31;43m Nmi:%.4f \033[0m" % nmi)
                    print("Rearrange:", "\033[1;31;43m Ari:%.4f \033[0m" % adjscore)
                    print('-' * 100)