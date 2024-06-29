import numpy as np
import scipy.sparse as sps


class MyGraph:
    def __init__(
            self,
            nodes=None,
            edges=None,
            node_subjects=None,
            *,
            is_directed=False,

    ):
        self.citations = None
        self.papers = None
        self.idx_map = None
        self.number_of_edges = None
        self.number_of_nodes = None
        self.nodes = nodes
        self.edges = edges


        self.is_directed = is_directed

        self.features = self.get_features(nodes)
        self.adj = self.make_adjacency_matrix(edges)
        self.node_subjects = node_subjects


        self.subjects_set = set(self.node_subjects.values)
        self.number_of_subjects = len(self.subjects_set)
        self.gatdata()


    def get_number_of_edges(self):
        return self.number_of_edges

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_number_of_feature(self):
        return len(self.features.columns)

    def get_adjacency_matrix(self):
        return self.adj

    def get_number_of_subjects(self):
        return self.number_of_subjects

    def get_node_subjects(self):
        return self.node_subjects

    def get_nodes(self):
        return self.nodes["paper"].index.values

    def get_features_dense_sparse(self):
        features_df = self.features.astype(dtype=np.dtype(str))
        features = sps.csr_matrix(features_df, dtype=np.float32).todense()
        return features

    def get_graph_info(self):
        print("number of nodes: " + str(self.number_of_nodes))
        print("number of edges: " + str(self.number_of_edges))

    def get_features(self, nodes):
        # return a Pandas DataFrame of features
        features_df = nodes["paper"]
        self.number_of_nodes = len(features_df)
        return features_df

    def nodes_to_loc(self, nodes_id):

        nodes_loc = np.array(list(map(self.idx_map.get, nodes_id)),
                         dtype=np.int32).reshape(nodes_id.shape)

        return nodes_loc.reshape(1, len(nodes_id))



    def make_adjacency_matrix(self, edges):
        edges_df = edges["cites"]
        nodes_order = self.nodes["paper"].index.values
        sources = edges_df.loc[:, "source"].values
        targets = edges_df.loc[:, "target"].values
        idx_map = {j: i for i, j in enumerate(nodes_order)}
        self.idx_map = idx_map

        nsources = np.array(list(map(idx_map.get, sources)),
                         dtype=np.int32).reshape(sources.shape)
        ntargets = np.array(list(map(idx_map.get, targets)),
                         dtype=np.int32).reshape(targets.shape)

        src_idx = nsources
        tgt_idx = ntargets
        self.number_of_edges = sources.size

        n = max(max(src_idx), max(tgt_idx)) + 1

        weights = np.ones(src_idx.shape)

        adj = sps.csr_matrix((weights, (tgt_idx, src_idx)), shape=(n, n), dtype=np.float32)

        if not self.is_directed and n > 0:
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            backward = adj.transpose(copy=True)
            (nonzero,) = backward.diagonal().nonzero()
            backward[nonzero, nonzero] = 0

            adj += backward

        adj.sum_duplicates()
        return adj

    def my_test(self):
        print("hello")

    def gatdata(self):
        self.citations = self.edges["cites"]

        subjects = self.node_subjects.values
        x = self.nodes["paper"]
        x['subject'] = subjects
        x = x.reset_index()
        x.rename(columns={'index': 'paper_id'}, inplace=True)
        x.rename(columns={'pid': 'paper_id'}, inplace=True)
        self.papers = x



