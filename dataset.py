import torch
import dgl
from dgl.data import DGLDataset
import os
import scipy.sparse as sp
import numpy as np
import json

class Tesseract(DGLDataset):
    def __init__(self, data_dir, feat_file, adj_mtx):
        self.data_dir = data_dir
        self.feat_mtx_file = feat_file
        self.adj_mtx_file = adj_mtx
        self.num_classes = 2
        super().__init__(name='Tesseract')

    def create_mask(self, num_nodes, idx_mask):
        if isinstance(idx_mask, dict):
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            for _, val in idx_mask.items():
                mask[val] = True
            return mask
        else:
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[idx_mask] = True
            return mask
        
    def process(self):
        adjacency_matrix = sp.load_npz(os.path.join(self.data_dir, self.adj_mtx_file))
        
        # scipy.sparse.csr.csr_matrix ==> scipy.sparse...coo_matrix ==> torch.sparse.Tensor
        node_feats = sp.load_npz(os.path.join(self.data_dir, self.feat_mtx_file))
        coo_feats = node_feats.tocoo()
        values = coo_feats.data
        indices = np.vstack((coo_feats.row, coo_feats.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_feats.shape
        sparse_feats = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        with open(os.path.join(self.data_dir, 'label_info','label_info.json'), 'r') as f:
            label_info = json.load(f)
        node_labels = []
        for _, _, label in label_info:
            node_labels.append(label)
        node_labels = np.array(node_labels, dtype=np.int64)
        node_labels = torch.from_numpy(node_labels)
        
        g = dgl.from_scipy(adjacency_matrix)
        g.ndata['features'] = sparse_feats
        g.ndata['labels'] = node_labels
        
        with open(os.path.join(self.data_dir, 'label_info', 'split_info.json'), 'r') as f:
            split_info = json.load(f)
        ids_train = split_info['train_ids']
        ids_val = split_info['val_ids']
        ids_test = split_info['test_ids']
        
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_feats.shape[0]
        train_mask = self.create_mask(n_nodes, ids_train)
        val_mask = self.create_mask(n_nodes, ids_val)

        # train_mask = self.create_mask(n_nodes, list(range(256)))
        # val_mask = self.create_mask(n_nodes, list(range(256, 320)))
        # ids_test = {
        #     "2015-1": list(range(320, 448)),
        #     "2015-2": list(range(448, 500)), 
        #     "2015-3": list(range(500, 600))
        # }

        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = self.create_mask(n_nodes, ids_test)
        for date, indices in ids_test.items():
            test_mask = "test_mask_" + date
            g.ndata[test_mask] = self.create_mask(n_nodes, indices)

        # self._g = dgl.to_bidirected(g)
        self._g = dgl.reorder_graph(g)

    def __getitem__(self, i):
        assert i == 0, "Only one graph in the dataset."
        return self._g

    def __len__(self):
        return 1