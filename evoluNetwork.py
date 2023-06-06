import os
import pickle as pkl
from tracemalloc import start
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

"""
    Evolutionary network
    Paper: GraphEvolveDroid: Mitigate Model Degradation in the Scenario of Android Ecosystem Evolution
    Source: https://github.com/liangxun/GraphEvolveDroid
"""

class EvolutionGraph:
    """
    Construct an evolutionary network proposed in paper.
    Evolutionary network is a KNN graph with timestamp constraints.
    """

    def __init__(self, feat, k, batch_size=5000):
        self.k = k

        # dimensionality reduction
        # Samples have been sorted by timestamp in preprocessing stage
        pca = TruncatedSVD(n_components=64)
        pca.fit(feat)
        self.feat = pca.transform(feat)
        self.batch_size = batch_size
        self.n = self.feat.shape[0]
    
    def get_adj(self):
        graph = list()
        
        # step 1: calculate cosine similarity in the first batch
        batch = self.feat[:self.batch_size]
        batch = batch.dot(batch.T)

        # step 2: select top K similar neighbors without timestamp constraints
        batch = self._init_get_top_k_neighbors(batch)
        graph.append(batch)

        # construct Evolutionary netowrk with timestamp constraints
        start_id = self.batch_size
        while start_id < self.n:
            end_id = min(start_id + self.batch_size, self.n)
            print((start_id, end_id))

            # step1: calculate cosine similarity between nodes and their parent nodes
            batch = self.feat[start_id: end_id]
            batch = batch.dot(self.feat[:end_id].T)

            # step2: select top K similar neighbors with timestamp constraints
            batch = self._get_top_k_neighbors(batch, start_id)
            start_id += self.batch_size
            graph.append(batch)

        return sp.vstack(graph)

    def _get_top_k_neighbors(self, batch, start_id):
        """
        Select top k similar neighbors, where node_idx is smaller than current node_idx
        """
        csr_row, csr_col, csr_data = list(), list(), list()

        for row, sim in enumerate(batch):
            idx = row + start_id
            cols = np.argpartition(sim[:idx], -self.k)[-self.k:]
            data = sim[cols]
            csr_row += self.k * [row]
            csr_col += cols.tolist()
            csr_data += data.tolist()

        return sp.csr_matrix((csr_data, (csr_row, csr_col)), shape=[batch.shape[0], self.n])

    def _init_get_top_k_neighbors(self, batch):
        """
        In the first batch, there is no timestamp constraints.
        """
        csr_row, csr_col, csr_data = list(), list(), list()

        for row, sim in enumerate(batch):
            cols = np.argpartition(sim, -self.k)[-self.k:]
            data = sim[cols]
            csr_row += self.k * [row]
            csr_col += cols.tolist()
            csr_data += data.tolist()

        return sp.csr_matrix((csr_data, (csr_row, csr_col)), shape=[batch.shape[0], self.n])

import torch
import json
import datetime
import itertools
from sklearn.preprocessing import normalize

class TransferGraph:

    def __init__(self, feat, split_info, k, gpu, batch_size=5000):
        self.k = k
        
        self.ids_train = split_info['train_ids']
        self.ids_val = split_info['val_ids']
        test_ids_dict = split_info['test_ids']
        self.ids_test = list()
        for _, indices in test_ids_dict.items():
            self.ids_test += indices

        # dimensionality reduction
        # samples have been sorted by timestamp in preprocessing stage
        pca = TruncatedSVD(n_components=64)
        pca.fit(feat)
        self.feat = pca.transform(feat)
        self.feat = normalize(self.feat, axis=1)

        self.batch_size = batch_size
        self.n = self.feat.shape[0]
        self.num_target = len(self.ids_test)
        
        if gpu >= 0 and torch.cuda.is_available():
            device = 'cuda:{}'.format(gpu)
        else:
            device = 'cpu'
        
        self.feat = torch.from_numpy(self.feat).to(device)
    
    def get_adj(self):
        num_source = self.n - self.num_target
        source_feat = self.feat[:num_source]
        target_feat = self.feat[num_source:]
        graph = list()
        
        # construct transfer knn graph in source domain
        start_id = self.batch_size
        while start_id < num_source:
            print(start_id)
            batch_feat = source_feat[start_id: min(start_id+self.batch_size, num_source)]

            # step 1: calculate cosine similarity in diff batch from source domain
            sim_mtx = batch_feat.matmul(source_feat[:start_id].T)

            # step 2: select top K similar neighbors for each target sample in source domain
            (values, indices) = sim_mtx.topk(self.k, dim=1, largest=True)
            
            # step3: construct a sparse matrix for each batch in source domain
            # the first batch should be batch_size-indexed, subsequent batches should be 0-indexed
            # to save memory, sp.csr_matrix((data, indices, indptr), shape=[x, y]) is used
            if start_id == self.batch_size: # process the first batch individually
                indptr = [0] * self.batch_size + [i * self.k for i in range(indices.shape[0] + 1)]
                dim0 = sim_mtx.shape[0] + self.batch_size
            else:
                indptr = [i * self.k for i in range(indices.shape[0] + 1)]
                dim0 = sim_mtx.shape[0]
            indices = list(itertools.chain(*indices.tolist()))
            data = list(itertools.chain(*values.tolist()))

            batch = sp.csr_matrix((data, indices, indptr), shape=[dim0, self.n])
            graph.append(batch)
            print(batch.shape)
            start_id += self.batch_size

        # construct transfer knn graph between source domain and target domain
        start = datetime.datetime.now()
        start_id = 0
        mini_batch_size = 256
        while start_id < self.num_target:
            batch_feat = target_feat[start_id: min(start_id+mini_batch_size, self.num_target)]

            # step 1: calculate cosine similarity between source domain samples and target domain samples
            sim_mtx = batch_feat.matmul(source_feat.T)

            # step 2: select top K similar neighbors for each target sample in source domain
            (values, indices) = sim_mtx.topk(self.k, dim=1, largest=True)
            
            indptr = [i * self.k for i in range(indices.shape[0] + 1)]
            indices = list(itertools.chain(*indices.tolist()))
            data = list(itertools.chain(*values.tolist()))

            batch = sp.csr_matrix((data, indices, indptr), shape=[sim_mtx.shape[0], self.n])
            graph.append(batch)
            start_id += mini_batch_size

        end = datetime.datetime.now()
        print("time: {}".format(end - start))
        
        return sp.vstack(graph)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="construct knn graph.")
    parser.add_argument('--keyword', type=str, default='drebin')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    args = parser.parse_args()

    # data_dir = "/home/sunrui/data/GraphEvolveDroid"
    data_dir = "/home/sunrui/data/drebin"
    feat = sp.load_npz(os.path.join(data_dir, "{}_feat.npz".format(args.keyword)))

    graph = EvolutionGraph(feat, args.k, batch_size=10000)
    adj_mtx = graph.get_adj()
    print(adj_mtx)
    sp.save_npz(os.path.join(data_dir, "{}_knn_{}.npz".format(args.keyword, args.k)), adj_mtx)
    sp.save_npz(os.path.join(data_dir, "{}_knn_{}_T.npz".format(args.keyword, args.k)), adj_mtx.transpose())

    # with open(os.path.join(data_dir, 'label_info', 'split_info.json'), 'r') as f:
    #     split_info = json.load(f)
    
    # graph = TransferGraph(feat, split_info, args.k, args.gpu, batch_size=5000)
    # adj_mtx = graph.get_adj()
    # print(adj_mtx.shape)
    # sp.save_npz(os.path.join(data_dir, "{}_rev_tf_knn_{}.npz".format(args.keyword, args.k)), adj_mtx)
    # sp.save_npz(os.path.join(data_dir, "{}_tf_knn_{}.npz".format(args.keyword, args.k)), adj_mtx.transpose())