import torch
import pickle
import pandas as pd
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import argparse
from utils.preprocessing import make_data
from utils.train_and_test import smiles_to_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_saved_data', action='store_true', default=False)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--relu', action='store_true', default=False)
    parser.add_argument('--task_targeted', default='delaney')
    parser.add_argument('--task_trained', default='qm9')
    parser.add_argument('--model', default='gcn')
    parser.add_argument('--percentage', default=0.10, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    # define args
    args = parse_args()
    if args.normalize:
        args.task_trained += '_normalize'
    if args.relu:
        args.task_trained += '_relu'
    if args.seed is not None:
        args.task_trained += '_' + args.seed
    task_str = args.task_targeted  # used for saving pickle
    args.task_targeted = args.task_targeted.split('+')

    # define featurizer
    atom_featurizer = CanonicalAtomFeaturizer()
    if args.model == 'gcn':
        edge_featurizer = None
        features_name = 'atom'
    else:
        edge_featurizer = CanonicalBondFeaturizer()
        features_name = 'atom_edge'

    # make data
    # data_dict['delaney'] = [train_dataframe, ..., train_loader, ...]
    data_dict = {}
    for task in args.task_targeted:
        # load data
        if args.use_saved_data:
            with open('save_pickle/' + task + '_' + features_name + '_' +
                      'eval_data.p', 'rb') as f:
                data_dict[task] = pickle.load(f)
        # save data
        else:
            data_dict[task] = make_data(task, atom_featurizer,
                                        edge_featurizer, batch_size=64,
                                        shuffle=False)
            with open('save_pickle/' + task + '_' + features_name + '_' +
                      'eval_data.p', 'wb') as f:
                pickle.dump(data_dict[task], f)

    # load model
    gnn_net = torch.load("save_pickle/" + args.task_trained + '_' + args.model
                         + "_model.p")

    for task in args.task_targeted:
        df = pd.concat([data_dict[task][0], data_dict[task][1],
                        data_dict[task][2]])
        # もしもtestだけで実験をしたい場合はtestのみを残す
        loaders = [data_dict[task][3], data_dict[task][4], data_dict[task][5]]
        task_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        smiles_to_embeddings(loaders, gnn_net, args.task_trained,
                             task, args.model, df, task_names,
                             percentage=args.percentage)
