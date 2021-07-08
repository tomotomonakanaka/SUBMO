import torch
import numpy as np
import pandas as pd
import pickle
from utils.algorithms import greedy_logdet_max, greedy_baseline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(loader, gnn_net, loss_fn, optimizer, task, model_name,
                    data_len, epoch):
    gnn_net.train()
    epoch_loss = 0
    for bg, labels in loader:
        atom_feats = bg.ndata.pop('h').to(device)
        if model_name == 'gcn':
            pred, _ = gnn_net(bg, atom_feats, task)
        else:
            edge_feats = bg.edata.pop('e').to(device)
            pred, _ = gnn_net(bg, atom_feats, task, edge_feats)
        labels = labels.reshape([labels.shape[0], -1])
        pred = pred.reshape([pred.shape[0], -1])
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item() * labels.shape[0]
    epoch_loss /= data_len
    if epoch % 10 == 0:
        print(f"epoch: {epoch}, task: {task}, LOSS: {epoch_loss:.3f}")


def eval_one_epoch(loader, gnn_net, loss_fn, task, model_name, data_len,
                   eval_dataset=None, percentage=0.1, task_str=None):
        gnn_net.eval()
        epoch_loss = 0
        preds = None
        trues = None
        embeddings = None
        for bg, labels in loader:
            atom_feats = bg.ndata.pop('h').to(device)
            if model_name == 'gcn':
                pred, embedding = gnn_net(bg, atom_feats, task)
            else:
                edge_feats = bg.edata.pop('e').to(device)
                pred, embedding = gnn_net(bg, atom_feats, task, edge_feats)
            labels = labels.reshape([labels.shape[0], -1])
            pred = pred.reshape([pred.shape[0], -1])
            loss = loss_fn(pred, labels)
            epoch_loss += loss.detach().item() * labels.shape[0]

            # prediction, ground-truth, embedding
            pred_cpu = pred.detach().to('cpu').numpy()
            true_cpu = labels.to('cpu').numpy()
            embedding_cpu = embedding.detach().to('cpu').numpy()
            if preds is None:
                preds = pred_cpu
                trues = true_cpu
                embeddings = embedding_cpu
            else:
                preds = np.append(preds, pred_cpu, axis=0)
                trues = np.append(trues, true_cpu, axis=0)
                embeddings = np.append(embeddings, embedding_cpu, axis=0)
        epoch_loss /= data_len
        print(f"valid {task} LOSS: {epoch_loss:.3f}")

        # save prediction
        if eval_dataset is not None:
            task_names = eval_dataset.columns.to_list()
            task_names.remove('smiles')
            task_names.remove('mols')
            pred_names = ['pred ' + name for name in task_names]
            eval_dataset[pred_names] = pd.DataFrame(preds)
            eval_dataset['embedding'] = list(embeddings)
            df = eval_dataset
            df = greedy_logdet_max(df, percentage)
            df = greedy_baseline(df, percentage, rule='maxsum')
            df = greedy_baseline(df, percentage, rule='maxmin')
            with open('save_pickle/' + task + '_' + model_name + '_' + task_str
                      + '_' + 'predicted_data.p', 'wb') as f:
                pickle.dump(df, f)
