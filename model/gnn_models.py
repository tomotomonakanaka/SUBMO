import torch
import torch.nn as nn
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgl.readout import max_nodes
from dgl.nn.pytorch import WeightAndSum


class TaskLayer(nn.Module):
    """ TaskLayer predict properties from molecular graph vectors.
    Parameters
    ----------
    in_feats : int
        Number of input molecular graph features
    hidden_feats : int
        Number of molecular graph features in hidden layers
    n_tasks : int
        Number of tasks, also output size
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(TaskLayer, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, n_tasks)
        )

    def forward(self, h):
        """
        Input
        ----------
        h : FloatTensor of shape (B, M)
            * B is the number of molecules in a batch
            * M is the input molecule feature size, must match in_feats
        Output
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(h)


class GNNMultitask(nn.Module):
    """
    Parameters
    ----------
    node_in_feats : int
        Number of input atom features
    edge_in_feats : int
        Number of input edge features
    n_tasks : dict
        Number of prediction tasks
        eg. {'delaney': 1}
    gnn_hidden_feats : list of int or int
        gnn_hidden_feats[i] gives the number of output atom features
        in the i+1-th gnn layer
    model_type : string
        the model type of gnn
        candidates are 'gcn', 'attentivefp'
    graph_hidden_feats : int
        Number of molecular graph features in hidden layers of the TaskLayer.
    dropout : float
        The probability for dropout in TaskLayer. Default to be 0., i.e. no
        dropout is performed.
    normalization : bool
        if normalization=True, hidden features are normalized during training and test.
    num_layers : int
        the number of layers in the first attentive layer
    num_timesteps : int
        the number of layers in the second attentive layer
    """
    def __init__(self, node_in_feats, n_tasks, gnn_hidden_feats=None,
                 edge_in_feats=None, model_type='gcn',
                 graph_hidden_feats=100, dropout=0., normalization=False,
                 num_layers=2, num_timesteps=2, relu=False):
        super(GNNMultitask, self).__init__()

        # parameters
        self.normalization = normalization
        self.relu = relu
        self.model_type = model_type
        if gnn_hidden_feats is None:
            gnn_hidden_feats = [graph_hidden_feats]
        self.weighted_sum_readout = WeightAndSum(gnn_hidden_feats[-1])

        # gnn models
        if model_type == 'gcn':
            self.gnn = GCN(node_in_feats, hidden_feats=gnn_hidden_feats)
            self.g_feats = 2 * gnn_hidden_feats[-1]
        elif model_type == 'attentivefp':
            self.gnn = AttentiveFPGNN(node_feat_size=node_in_feats,
                                      edge_feat_size=edge_in_feats,
                                      num_layers=num_layers,
                                      graph_feat_size=graph_hidden_feats,
                                      dropout=dropout)
            self.readout = AttentiveFPReadout(feat_size=graph_hidden_feats,
                                              num_timesteps=num_timesteps,
                                              dropout=dropout)
            self.g_feats = gnn_hidden_feats[-1]

        # layers for tasks
        self.soft_classifier = nn.ModuleDict()
        for task in n_tasks:
            self.soft_classifier[task] = TaskLayer(
                self.g_feats, int(self.g_feats/2), n_tasks[task], dropout)

    def forward(self, g, node_feats, task=None, edge_feats=None):
        if edge_feats is None:
            feats = self.gnn(g, node_feats)
            # readout function
            h_g_sum = self.weighted_sum_readout(g, feats)

            with g.local_scope():
                g.ndata['h'] = feats
                h_g_max = max_nodes(g, 'h')

            h_g = torch.cat([h_g_sum, h_g_max], dim=1)  # shape (n, g_feats)

        else:
            feats = self.gnn(g, node_feats, edge_feats)
            h_g, node_weights = self.readout(g, feats, True)  # node weight for analysis

        if self.relu:
            h_g = nn.functional.relu(h_g)

        if self.normalization:
            h_g = (h_g.T / torch.norm(h_g, dim=1)).T

        # if task is None we don't need output
        if task is not None:
            output = self.soft_classifier[task](h_g)
            return output, h_g

        else:
            return h_g
