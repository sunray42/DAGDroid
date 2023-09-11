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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="construct knn graph.")
    parser.add_argument('--keyword', type=str, default='drebin')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    args = parser.parse_args()

    data_dir = ""
    feat = sp.load_npz(os.path.join(data_dir, "{}_feat_mtx.npz".format(args.keyword)))

    graph = EvolutionGraph(feat, args.k, batch_size=10000)
    adj_mtx = graph.get_adj()
    print(adj_mtx)
    # sp.save_npz(os.path.join(data_dir, "{}_knn_{}.npz".format(args.keyword, args.k)), adj_mtx)