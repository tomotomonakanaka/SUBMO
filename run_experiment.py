import random
import pickle
import argparse
import numpy as np
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from model.gnn_models import GNNMultitask
from utils.preprocessing import make_data
from utils.train_and_test import train_one_epoch, eval_one_epoch


def parse_args():
    '''
    seed: random seed e.g. 1
    use_saved_data: if true, this program will use preprocessed data
    save_model: if true, the trained model will be saved
    tasks: the names of tasks used e.g. delaney+qm9
    normalize: if true, molecular vectors will be normalized
    relu: if true, molecular vectors will have positive values.
    model: name of the model used e.g. attentivefp or gcn
    percentage: the percentage of molecules which will be selected
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None)
    parser.add_argument('--use_saved_data', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--tasks', default='delaney')
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--relu', action='store_true', default=False)
    parser.add_argument('--model', default='attentivefp')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--T', default=2, type=int)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--percentage', default=0.1, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    # check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    # define args
    args = parse_args()
    task_str = args.tasks  # used for saving pickle
    if args.normalize:
        task_str += '_normalize'
    if args.relu:
        task_str += '_relu'
    if args.seed is not None:
        task_str += '_' + args.seed
    args.tasks = args.tasks.split('+')

    # seed
    random_seed = int(args.seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device == 'cuda':
        torch.cuda.manualseed_all(random_seed)

    # define featurizer
    atom_featurizer = CanonicalAtomFeaturizer()
    if args.model == 'gcn':
        edge_featurizer = None
        features_name = 'atom'
    else:
        edge_featurizer = CanonicalBondFeaturizer()
        features_name = 'atom_edge'

    # define node and edge dim
    n_feats = atom_featurizer.feat_size()
    if args.model != "gcn":
        e_feats = edge_featurizer.feat_size()

    # make data
    # data_dict['delaney'] = [train_dataframe, ..., train_loader, ...]
    data_dict = {}
    for task in args.tasks:
        # load data
        if args.use_saved_data:
            with open('save_pickle/' + task + '_' + features_name + '_' +
                      'data.pickle', 'rb') as f:
                data_dict[task] = pickle.load(f)
        # save data
        else:
            data_dict[task] = make_data(task, atom_featurizer, edge_featurizer,
                                        batch_size=args.batch_size)
            with open('save_pickle/' + task + '_' + features_name + '_' +
                      'data.pickle', 'wb') as f:
                pickle.dump(data_dict[task], f)

    # task dim
    # ncls = {'delaney':1, ...}
    ncls = {}
    for task in args.tasks:
        for bg, label in data_dict[task][3]:
            ncls[task] = label.shape[-1]
            break

    # define gnn
    if args.model == 'gcn':
        gnn_net = GNNMultitask(node_in_feats=n_feats,
                               n_tasks=ncls,
                               gnn_hidden_feats=[60, 20],
                               graph_hidden_feats=20,
                               dropout=args.dropout,
                               normalization=args.normalize).to(device)
    else:
        gnn_net = GNNMultitask(node_in_feats=n_feats,
                               edge_in_feats=e_feats,
                               n_tasks=ncls,
                               model_type=args.model,
                               dropout=args.dropout,
                               graph_hidden_feats=args.dim,
                               normalization=args.normalize,
                               relu = args.relu,
                               num_layers=args.radius,
                               num_timesteps=args.T).to(device)

    # define optimizer and loss
    optimizer = torch.optim.Adam(gnn_net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # train
    for epoch in range(1, args.epoch + 1):
        for task in args.tasks:
            train_one_epoch(data_dict[task][3], gnn_net, loss_fn, optimizer,
                            task, args.model, len(data_dict[task][0]), epoch)
            if epoch % 10 == 0:
                eval_one_epoch(data_dict[task][4], gnn_net, loss_fn, task,
                               args.model, len(data_dict[task][1]))

    # save model
    if args.save_model:
        torch.save(gnn_net, "save_pickle/" + task_str + '_' + args.model
                   + "_model.pickle")

    # evaluation
    for task in args.tasks:
        eval_one_epoch(data_dict[task][5], gnn_net, loss_fn, task, args.model,
                       len(data_dict[task][2]), data_dict[task][2],
                       percentage=args.percentage, task_str=task_str)
