import numpy as np
import pandas as pd
import dgl
import torch
import deepchem as dc
from rdkit import Chem
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from utils.calc_chem import get_natom, get_nelec


def graph_featurizer(smiles, atom_featurizer, edge_featurizer, with_H):
    # convert mols to graphs with feature
    if edge_featurizer is None:
        graphs = [smiles_to_bigraph(smile, add_self_loop=True,
                  node_featurizer=atom_featurizer,
                  explicit_hydrogens=with_H)
                  for smile in smiles]
    else:
        graphs = [smiles_to_bigraph(smile,
                  node_featurizer=atom_featurizer,
                  edge_featurizer=edge_featurizer,
                  explicit_hydrogens=with_H)
                  for smile in smiles]
    graphs_dropna = list(filter(None, graphs))
    return graphs_dropna


def make_dataframe(dataset):
    # return dataframe with graphs and y
    dataframe = pd.DataFrame(dataset.y, columns=dataset.get_task_names())
    mols = [Chem.MolFromSmiles(m) for m in dataset.ids]
    dataframe['mols'] = mols
    dataframe['smiles'] = dataset.ids
    dataframe_dropna = dataframe.dropna(how='any')
    print(len(dataframe)-len(dataframe_dropna),
          "smiles cannot be converted to mols in", len(dataframe))
    return dataframe_dropna


def make_qm9_dataframe(divide=False):
    tasks = [
       "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298",
       "h298", "g298", "cv"
    ]

    # read dataset
    raw_filename = "data/qm9.csv"
    qm9_df = pd.read_csv(raw_filename)
    smilesList = qm9_df.smiles.values

    # canonical smiles
    remained_smiles = []
    canonical_smiles_list = []
    mols = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles), isomericSmiles=True))
            mols.append(mol)
        except:
            print(smiles)
            pass
    qm9_df = qm9_df[qm9_df["smiles"].isin(remained_smiles)]
    qm9_df['cano_smiles'] = canonical_smiles_list
    qm9_df['mols'] = mols
    print(len(smilesList)-len(qm9_df),
          "smiles cannot be converted to mols in", len(smilesList))

    # normalize the values
    df = pd.DataFrame()
    for task in tasks:
        df[task] = qm9_df[task]
    df['mols'] = qm9_df.mols
    df['smiles'] = qm9_df.cano_smiles

    if divide==True:
        Nelecs = df['mols'].apply(get_nelec)
        df['zpve'] = df.apply(lambda x: x['zpve'] / (3 * get_natom(x['mols']) - 6), axis = 1)
        df['u0'] = df['u0'] / Nelecs
        df['u298'] = df['u298'] / Nelecs
        df['h298'] = df['h298'] / Nelecs
        df['g298'] = df['g298'] / Nelecs
        df['cv'] = df.apply(lambda x: x['cv'] / (3 * get_natom(x['mols']) - 6), axis = 1)

    # train, valid, test split
    random_seed = 888
    test_df = df.sample(frac=1/10, random_state=random_seed)
    training_data = df.drop(test_df.index)
    valid_df = training_data.sample(frac=1/9, random_state=random_seed)
    train_df = training_data.drop(valid_df.index)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for task in tasks:
        mean = train_df[task].mean()
        std = train_df[task].std()
        train_df[task] = (train_df[task] - mean)/std
        valid_df[task] = (valid_df[task] - mean)/std
        test_df[task] = (test_df[task] - mean)/std

    return train_df, valid_df, test_df


def collate(sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    return batched_graph.to(device), torch.tensor(labels).to(device)


def make_dataloader(dataframe, task_name, graphs, batch_size=128,
                    shuffle=False, drop_last=False):
    y = dataframe[task_name].to_numpy(dtype=np.float32)
    data = list(zip(graphs, y))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate, drop_last=drop_last)
    return dataloader


def make_data(arg_tasks, atom_featurizer, edge_featurizer, batch_size=128,
              shuffle=True):
    with_H = False
    if arg_tasks == 'delaney':  # ESOL regression
        tasks, datasets, transformers = dc.molnet.load_delaney(reload=False)
    if arg_tasks == 'sampl':  # FreeSolv regression
        tasks, datasets, transformers = dc.molnet.load_sampl(reload=False)
    if arg_tasks == 'lipo':  # Lipop regression
        tasks, datasets, transformers = dc.molnet.load_lipo(reload=False)

    if arg_tasks == 'qm9':
        train_dataframe, valid_dataframe, test_dataframe = make_qm9_dataframe()
    elif arg_tasks == 'qm9_divide':
        train_dataframe, valid_dataframe, test_dataframe = make_qm9_dataframe(divide=True)
    else:
        train_dataset, valid_dataset, test_dataset = datasets
        # make dataframe
        train_dataframe = make_dataframe(train_dataset)
        valid_dataframe = make_dataframe(valid_dataset)
        test_dataframe = make_dataframe(test_dataset)

    # convert mols to graphs with feature
    train_graphs = graph_featurizer(train_dataframe.smiles, atom_featurizer,
                                    edge_featurizer, with_H)
    valid_graphs = graph_featurizer(valid_dataframe.smiles, atom_featurizer,
                                    edge_featurizer, with_H)
    test_graphs = graph_featurizer(test_dataframe.smiles, atom_featurizer,
                                   edge_featurizer, with_H)

    if len(train_graphs) != len(train_dataframe):
        print("**********ERROR**********")
        print("the length of dataframe and graphs are different")

    # task name
    task_names = train_dataframe.columns.to_list()
    task_names.remove('smiles')
    task_names.remove('mols')

    # make loader
    train_loader = make_dataloader(train_dataframe, task_names, train_graphs,
                                   batch_size=batch_size, shuffle=shuffle,
                                   drop_last=False)
    valid_loader = make_dataloader(valid_dataframe, task_names, valid_graphs,
                                   batch_size=batch_size)
    test_loader = make_dataloader(test_dataframe, task_names, test_graphs,
                                  batch_size=batch_size)

    return train_dataframe, valid_dataframe, test_dataframe, \
           train_loader, valid_loader, test_loader
