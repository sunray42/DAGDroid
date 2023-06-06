import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, activation, dropout, agg='mean'):
        super(SAGE, self).__init__()
        self.init(in_size, hid_size, out_size, n_layers, activation, dropout, agg)

    def init(self, in_size, hid_size, out_size, n_layers, activation, dropout, agg):
        self.n_layers = n_layers
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, agg))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, agg))
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, agg))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, agg))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = self.activation(h)
            h = self.dropout(h)
        return h

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self.out_size

    def load_subtensor(self, nfeat, input_nodes, device):
        """
        Extracts features for a subset of nodes
        """
        if nfeat.is_sparse:
            batch_inputs = [nfeat[node].to_dense() for node in input_nodes]
            batch_inputs = torch.stack(batch_inputs).to(device)
        else:
            batch_inputs = nfeat[input_nodes].to(device)
        return batch_inputs

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['features']
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g,
                                torch.arange(g.num_nodes()).to(g.device),
                                sampler,
                                device=device,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(),
                            self.hid_size if l != len(self.layers) - 1 else self.out_size,
                            device=buffer_device,
                            pin_memory=pin_memory)
            for input_nodes, output_nodes, blocks in dataloader:
                x = self.load_subtensor(feat, input_nodes, device)
                h = layer(blocks[0], x) # len(blocks) = 1
                h = self.activation(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y.to(device)

class MyClassifier(nn.Module):
    def __init__(self, backbone, num_classes, head=None, finetune=True):
        super(MyClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self._features_dim = backbone.out_features
        self.finetune = finetune
        
        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
    
    @property
    def features_dim(self):
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, blocks, x):
        """"""
        f = self.backbone(blocks, x)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

    def inference(self, g, device, batch_size):
        f = self.backbone.inference(g, device, batch_size)
        return self.head(f)