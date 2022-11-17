from utils import *


class MotifCounter:
    def __init__(self, target_motif, dataset_name, simple_relational_digraphs, cache_dir, logger=None, verbose=True):
        self.verbose = verbose
        self.dataset_name = dataset_name
        # self.target_motifs = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13')
        self.target_motifs = (target_motif,)
        self.logger = logger
        self.cache_dir = cache_dir
        self.num_node = simple_relational_digraphs[0].shape[0]
        self.num_edge = simple_relational_digraphs[0].nnz
        row, col = simple_relational_digraphs[0].nonzero()
        self.edge_index = list(zip(row.tolist(), col.tolist()))
        ckg_adj = sum(simple_relational_digraphs)
        ckg_bi = ckg_adj.multiply(ckg_adj.transpose())
        self.ckg_csr = sp.csr_matrix(ckg_adj) + sp.eye(self.num_node)
        self.ckg_csc = sp.csc_matrix(ckg_adj)
        self.ckg_bi = ckg_bi.tocsr()
        self.raw_graph = ckg_adj.tocoo()
        self.raw_bi_graph = ckg_bi.tocoo()

        self.ckg_bi = ckg_bi.tocsr()
        src, dst = ckg_adj.nonzero()
        self.edges_from_raw_graph = list(zip(src.tolist(), dst.tolist()))
        src, dst = ckg_bi.nonzero()
        self.bi_edges_from_raw_graph = list(zip(src.tolist(), dst.tolist()))

    def split_13motif_adjs(self):
        motif_adjs = []

        t = time.time()
        # edge_motif_weight = torch.zeros(train_graph.number_of_edges(), len(self.target_motifs))
        U = self.raw_graph
        B = self.raw_bi_graph

        for i, motif_name in enumerate(self.target_motifs):
            if self.verbose:
                print(f"Begin count {motif_name}-motif weight for each edge...")
            if motif_name == 'M1':
                if self.dataset_name in ['amazon-book']:
                    if os.path.exists(self.cache_dir + 'uuut.h5'):
                        fr = tb.open_file(self.cache_dir + 'uuut.h5', 'r')
                        C = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()
                    else:
                        C = tb_matmul_and_multiply(U, U, U.transpose(), self.cache_dir + 'uuut.h5')
                else:
                    C = U.dot(U).multiply(U.transpose())
                # C = U.dot(U).multiply(U.transpose())
                motif_adj = C + C.transpose()

            elif motif_name == 'M2':
                if self.dataset_name in ['amazon-book']:
                    if os.path.exists(self.cache_dir + 'buut.h5'):
                        fr = tb.open_file(self.cache_dir + 'buut.h5', 'r')
                        C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'ubut.h5', 'r')
                        C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'uub.h5', 'r')
                        C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        C = C1 + C2 + C3
                    else:
                        C1 = tb_matmul_and_multiply(B, U, U.transpose(), self.cache_dir + 'buut.h5')
                        C2 = tb_matmul_and_multiply(U, B, U.transpose(), self.cache_dir + 'ubut.h5')
                        C3 = tb_matmul_and_multiply(U, U, B, self.cache_dir + 'uub.h5')

                        C = C1 + C2 + C3
                else:
                    C = B.dot(U).multiply(U.transpose()) + U.dot(B).multiply(U.transpose()) + U.dot(U).multiply(B)

                motif_adj = C + C.transpose()

            elif motif_name == 'M3':
                if self.dataset_name in ['amazon-book']:
                    if os.path.exists(self.cache_dir + 'bbu.h5'):
                        fr = tb.open_file(self.cache_dir + 'bbu.h5', 'r')
                        C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'bub.h5', 'r')
                        C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'ubb.h5', 'r')
                        C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        C = C1 + C2 + C3
                    else:
                        C1 = tb_matmul_and_multiply(B, B, U, self.cache_dir + 'bbu.h5')
                        C2 = tb_matmul_and_multiply(B, U, B, self.cache_dir + 'bub.h5')
                        C3 = tb_matmul_and_multiply(U, B, B, self.cache_dir + 'ubb.h5')

                        C = C1 + C2 + C3
                else:
                    C = B.dot(B).multiply(U) + B.dot(U).multiply(B) + U.dot(B).multiply(B)
                motif_adj = C + C.transpose()

            elif motif_name == 'M4':
                if self.dataset_name in ['amazon-book']:
                    if os.path.exists(self.cache_dir + 'bbb.h5'):
                        fr = tb.open_file(self.cache_dir + 'bbb.h5', 'r')
                        motif_adj = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()
                    else:
                        motif_adj = tb_matmul_and_multiply(B, B, B, self.cache_dir + 'bbb.h5')
                else:
                    motif_adj = B.dot(B).multiply(B)

            elif motif_name == 'M5':
                # C = U.dot(U).multiply(U) + U.dot(U.transpose()).multiply(U) + U.transpose().dot(U).multiply(U)
                # C1 = U.dot(U).multiply(U) + U.transpose().dot(U).multiply(U)
                if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b']:
                    if os.path.exists(self.cache_dir + 'uuu.h5'):
                        fr = tb.open_file(self.cache_dir + 'uuu.h5', 'r')
                        C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'uutu.h5', 'r')
                        C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'utuu.h5', 'r')
                        C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        C = C1 + C2 + C3
                    else:
                        C1 = tb_matmul_and_multiply(U, U, U, self.cache_dir + 'uuu.h5')
                        C2 = tb_matmul_and_multiply(U, U.transpose(), U, self.cache_dir + 'uutu.h5')
                        C3 = tb_matmul_and_multiply(U.transpose(), U, U, self.cache_dir + 'utuu.h5')

                        C = C1 + C2 + C3
                else:
                    C = U.dot(U).multiply(U) + U.dot(U.transpose()).multiply(U) + U.transpose().dot(U).multiply(U)

                motif_adj = C + C.transpose()

            elif motif_name == 'M6':
                # U = convert_sparse_matrix_to_sparse_tensor(U).to(device)
                # B = convert_sparse_matrix_to_sparse_tensor(B).to(device)
                # motif_adj = U.dot(B) * U + B.dot(U.t()) * U.t() + U.t().dot(U) * B
                if self.dataset_name in ['amazon-book', 'yelp2018']:
                    if os.path.exists(self.cache_dir + 'ubu.h5'):
                        fr = tb.open_file(self.cache_dir + 'ubu.h5', 'r')
                        C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'butut.h5', 'r')
                        C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'utub.h5', 'r')
                        C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        motif_adj = C1 + C2 + C3
                    else:
                        C1 = tb_matmul_and_multiply(U, B, U, self.cache_dir + 'ubu.h5')
                        C2 = tb_matmul_and_multiply(B, U.transpose(), U.transpose(), self.cache_dir + 'butut.h5')
                        C3 = tb_matmul_and_multiply(U.transpose(), U, B, self.cache_dir + 'utub.h5')
                        motif_adj = C1 + C2 + C3
                else:
                    motif_adj = U.dot(B).multiply(U) + B.dot(U.transpose()).multiply(U.transpose()) + U.transpose().dot(U).multiply(B)

                # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                # neighbors, center_nodes = motif_adj.nonzero()
                # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

            elif motif_name == 'M7':
                # C1 = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U)
                if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b']:
                    if os.path.exists(self.cache_dir + 'utbut.h5'):
                        fr = tb.open_file(self.cache_dir + 'utbut.h5', 'r')
                        C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'buu.h5', 'r')
                        C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        fr = tb.open_file(self.cache_dir + 'uutb.h5', 'r')
                        C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                        fr.close()

                        motif_adj = C1 + C2 + C3
                    else:
                        C1 = tb_matmul_and_multiply(U.transpose(), B, U.transpose(), self.cache_dir + 'utbut.h5')
                        C2 = tb_matmul_and_multiply(B, U, U, self.cache_dir + 'buu.h5')
                        C3 = tb_matmul_and_multiply(U, U.transpose(), B, self.cache_dir + 'uutb.h5')
                        motif_adj = C1 + C2 + C3
                else:
                    motif_adj = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U) + U.dot(U.transpose()).multiply(B)

            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
                    motif_adj = sp.csr_matrix((self.num_node, self.num_node))
                    for center_node in range(self.num_node):
                        center_indices, out_neighbors = self.ckg_csr[center_node].nonzero()
                        out_neighbors = out_neighbors.tolist()

                        in_neighbors, center_indices = self.ckg_csc[:, center_node].nonzero()
                        in_neighbors = in_neighbors.tolist()

                        center_indices, bi_neighbors = self.ckg_bi[center_node].nonzero()
                        bi_neighbors = bi_neighbors.tolist()

                        pure_in_neighbors = list(set(in_neighbors) - set(bi_neighbors))
                        pure_out_neighbors = list(set(out_neighbors) - set(bi_neighbors))

                        if motif_name == 'M8':
                            if len(out_neighbors) > 0:
                                out_weights = len(out_neighbors) - 1

                                motif_adj[out_neighbors, [center_node]] += out_weights

                        elif motif_name == 'M9':
                            if len(bi_neighbors) > 0:
                                bi_weights = len(out_neighbors) + len(in_neighbors)

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_out_neighbors) > 0:
                                pure_out_weights = len(in_neighbors)

                                motif_adj[pure_out_neighbors, [center_node]] += pure_out_weights

                            if len(pure_in_neighbors) > 0:
                                pure_in_weights = len(out_neighbors)

                                motif_adj[pure_in_neighbors, [center_node]] += pure_in_weights

                        elif motif_name == 'M10':
                            if len(in_neighbors) > 0:
                                in_weights = len(in_neighbors) - 1

                                motif_adj[in_neighbors, [center_node]] += in_weights

                        elif motif_name == 'M11':
                            if len(bi_neighbors) > 0:
                                bi_weights = len(out_neighbors) - 1

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_out_neighbors) > 0:
                                pure_out_weights = len(bi_neighbors)

                                motif_adj[pure_out_neighbors, [center_node]] += pure_out_weights

                        elif motif_name == 'M12':
                            if len(bi_neighbors) > 0:
                                bi_weights = len(in_neighbors) - 1

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_in_neighbors) > 0:
                                pure_in_weights = len(bi_neighbors)

                                motif_adj[pure_in_neighbors, [center_node]] += pure_in_weights

                        elif motif_name == 'M13':
                            if len(bi_neighbors) > 0:
                                bi_weights = len(bi_neighbors) - 1

                                motif_adj[bi_neighbors, [center_node]] += bi_weights
                    motif_adj = motif_adj + motif_adj.transpose()

            # save(os.path.join(cache_path, motif_adj_file.format(i + 1)), motif_adj)
            motif_adjs.append(motif_adj)

        if self.verbose:
            print(f"The {len(self.target_motifs)} motif weight count process took {time.time() - t:.2f}s.")
        # torch.save(edge_motif_weight, self.cache_dir + edge_weight_file)

        edge_weight_sum = []
        for motif_adj in motif_adjs:
            edge_weight_sum.append(motif_adj.sum())

        num_motifs = {}
        for i, motif_name in enumerate(self.target_motifs):
            if int(motif_name[1:]) < 8:
                num_motifs[motif_name] = int(edge_weight_sum[i] / 6)
            else:
                num_motifs[motif_name] = int(edge_weight_sum[i] / 2)
                # assert edge_weight_sum[i] % 2 == 0

        info = [f"{motif_name}: {num_motifs[motif_name]} |" for motif_name in num_motifs]
        if self.logger is not None:
            self.logger.info(f"=====* Motifs included in {self.dataset_name} dataset *=====")
            self.logger.info(" ".join(info))
        else:
            if self.verbose:
                print(f"=====* Motifs included in {self.dataset_name} dataset *=====")
                print(" ".join(info))
        del num_motifs

        return motif_adjs


class MotifGraph(object):
    r"""Converts a graph to its corresponding M1-graph:

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, target_motif, dataset_name, cache_dir='', force_directed=False):
        self.target_motif = target_motif
        self.dataset_name = dataset_name
        self.force_directed = force_directed
        self.cache_dir = cache_dir

    def __call__(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
        row, col = edge_index[0], edge_index[1]

        sp_mat = sp.coo_matrix((np.ones_like(row), (row.tolist(), col.tolist())), shape=(N, N))
        simple_relational_digraphs = [sp_mat]
        mc = MotifCounter(self.target_motif, self.dataset_name, simple_relational_digraphs, self.cache_dir, verbose=False)
        sp_adj = mc.split_13motif_adjs()[0]
        sp_adj_norm = normalize_adj(sp_adj)
        edge_index = torch.tensor(sp_adj_norm.nonzero(), dtype=torch.long)
        # print("--***--", edge_index)
        data.edge_index = edge_index
        data.edge_weight4motif = torch.tensor(sp_adj_norm.data.reshape(-1, 1), dtype=torch.float32)
        data.x = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ScaledNormalize(object):
    def __call__(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
        row, col = edge_index[0], edge_index[1]
        sp_mat = sp.coo_matrix((np.ones_like(row), (row.tolist(), col.tolist())), shape=(N, N))
        sp_adj_norm = normalize_adj(sp_mat)
        data.edge_weight4norm = torch.tensor(sp_adj_norm.data.reshape(-1, 1), dtype=torch.float32)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConcatDataset(torch.utils.data.Dataset):
    """
    https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = datasets  # dataset list
        # print(self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
