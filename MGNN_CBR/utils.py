import numbers
import itertools
import math
import scipy.sparse as sp
import pickle
import numpy as np
import os
import time
import warnings

from tqdm import tqdm
import tables as tb
from scipy.sparse.linalg.eigen.arpack import eigsh

import os.path as osp
from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.data.data import Data


def repeat(src, length):
    if src is None:
        return None
    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def convert_sparse_tensor_to_sparse_matrix(sp_tensor):
    sp_tensor = sp_tensor.coalesce()
    data = sp_tensor.values().numpy()
    indices = sp_tensor.indices().numpy()
    row = indices[0, :]
    col = indices[1, :]
    return sp.coo_matrix((data, (row, col)), tuple(sp_tensor.shape))


def load(path):
    if 'pkl' in path:
        with open(path, 'rb') as fr:
            data = pickle.load(fr)
        return data
    elif 'npy' in path:
        return np.load(path, allow_pickle=False)
    else:
        with open(path) as fw:
            data = fw.readlines()
        return data


def save(path, data):
    if 'pkl' in path:
        with open(path, 'wb') as fw:
            pickle.dump(data, fw)
    elif 'npy' in path:
        np.save(path, data)
    else:
        with open(path, 'w') as fw:
            fw.writelines(data)


def tb_matmul_and_multiply(sp_mat1, sp_mat2, sp_mat3, hdf5_name, chunk_size=10000):
    csc_mat2 = sp_mat2.tocsc()
    csc_mat3 = sp_mat3.tocsc()
    # print(f"chunk_size={chunk_size}")
    left_dim, middle_dim, right_dim = sp_mat1.shape[0], sp_mat1.shape[1], sp_mat2.shape[1]
    fw = tb.open_file(hdf5_name, 'w')
    filters = tb.Filters(complevel=5, complib='blosc')
    row = fw.create_earray(fw.root, 'row', tb.Int32Atom(), shape=(0,), filters=filters)
    col = fw.create_earray(fw.root, 'col', tb.Int32Atom(), shape=(0,), filters=filters)
    data = fw.create_earray(fw.root, 'data', tb.Int32Atom(), shape=(0,), filters=filters)

    for i in tqdm(range(0, right_dim, chunk_size)):
        res = sp_mat1.dot(csc_mat2[:, i:min(i + chunk_size, right_dim)]).multiply(csc_mat3[:, i:min(i + chunk_size, right_dim)])
        data_i = res.data
        row_i, col_i = res.nonzero()
        data.append(data_i)
        row.append(row_i)
        col.append(i + col_i)

    product = sp.coo_matrix((fw.root.data[:], (fw.root.row[:], fw.root.col[:])), shape=sp_mat1.shape)
    fw.close()

    return product


def check_dir(directory_name):
    """

    :param directory_name: e.g. ./process_files or ../data/process_files
    :return:
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


class MotifCounter:
    def __init__(self, dataset_name, simple_relational_digraphs, cache_dir, logger=None):
        self.dataset_name = dataset_name
        self.target_motifs = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13')
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
        cache_path = os.path.join(self.cache_dir, f'motif_adj4{self.dataset_name}')
        check_dir(cache_path)
        motif_adj_file = 'm{}_adj4' + self.dataset_name + '.pkl'
        motif_adjs = []
        if os.path.exists(os.path.join(cache_path, motif_adj_file.format(1))):
            for i in range(13):
                motif_adjs.append(load(os.path.join(cache_path, motif_adj_file.format(i+1))))
        else:
            t = time.time()
            U = self.raw_graph
            B = self.raw_bi_graph

            for i, motif_name in enumerate(self.target_motifs):
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
                    if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b', 'chem2bio2rdf']:
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
                    if self.dataset_name in ['amazon-book', 'yelp2018', 'chem2bio2rdf']:
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

                elif motif_name == 'M7':
                    # C1 = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U)
                    if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b', 'chem2bio2rdf']:
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

                else:  # open motif
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
                        motif_adj = sp.csr_matrix((self.num_node, self.num_node))
                        for center_node in tqdm(range(self.num_node)):
                            center_indices, out_neighbors = self.ckg_csr[center_node].nonzero()  # 95595

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
                                    bi_weights = len(out_neighbors) + len(bi_neighbors) - 2

                                    motif_adj[bi_neighbors, [center_node]] += bi_weights

                                if len(pure_out_neighbors) > 0:
                                    pure_out_weights = len(bi_neighbors)

                                    motif_adj[pure_out_neighbors, [center_node]] += pure_out_weights

                            elif motif_name == 'M12':
                                if len(bi_neighbors) > 0:
                                    bi_weights = len(bi_neighbors) + len(in_neighbors) - 2

                                    motif_adj[bi_neighbors, [center_node]] += bi_weights

                                if len(pure_in_neighbors) > 0:
                                    pure_in_weights = len(bi_neighbors)

                                    motif_adj[pure_in_neighbors, [center_node]] += pure_in_weights

                            elif motif_name == 'M13':
                                if len(bi_neighbors) > 0:
                                    bi_weights = len(bi_neighbors) - 1

                                    motif_adj[bi_neighbors, [center_node]] += bi_weights
                        motif_adj = motif_adj + motif_adj.transpose()

                save(os.path.join(cache_path, motif_adj_file.format(i + 1)), motif_adj)
                motif_adjs.append(motif_adj)

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

        info = [f"{motif_name}: {num_motifs[motif_name]} |" for motif_name in num_motifs]
        if self.logger is not None:
            self.logger.info(f"=====* Motifs included in {self.dataset_name} dataset *=====")
            self.logger.info(" ".join(info))
        else:
            print(f"=====* Motifs included in {self.dataset_name} dataset *=====")
            print(" ".join(info))
        del num_motifs

        return motif_adjs

    def get_motif_weight(self, motif_mats):
        weight_file = os.path.join(self.cache_dir, 'motif_weight.pt')
        if os.path.exists(weight_file):
            motif_weight = torch.load(weight_file)
            print("motif weight file loaded.")
        else:
            motif_weight = torch.zeros(self.num_edge, 13)
            for i, motif_mat in enumerate(motif_mats):
                data = motif_mat.data.tolist()
                for j, node_pair in enumerate(zip(*motif_mat.nonzero())):
                    eid = self.edge_index.index(node_pair)
                    motif_weight[eid, i] += data[j]

            torch.save(motif_weight, weight_file)
            print("motif weight file saved.")

        return motif_weight


def convert_sparse_matrix_to_th_sparse_tensor(sp_mat):
    sp_mat = sp_mat.tocoo()
    indices = torch.LongTensor([sp_mat.row.tolist(), sp_mat.col.tolist()])
    data = torch.FloatTensor(sp_mat.data.tolist())
    size = torch.Size(sp_mat.shape)

    return torch.sparse.FloatTensor(indices, data, size).coalesce()


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]


def normalize_adj(sp_mat):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rowsum = np.array(np.sum(sp_mat, axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).reshape([-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        support = sp_mat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        I = sp.eye(sp_mat.shape[0])
        L = I - support
        L = L - ((lmax(L) / 2) * I)

    return L


class Chem2Bio2RDF(InMemoryDataset):
    r"""The richly heterogeneous dataset (biomedical knowledge graph) integrating data from multiple
    public sources spanning biomedical sub-domains including bioinformatics, cheminformatics and
    chemical biology from the
    "RCesheaercmh ar2ticBleio2RDF: a semantic framework for linking and data mining chemogenomic
    and systems chemical biology data" and the
    "edge2vec: Representation learning using edge semantics for biomedical knowledge discovery"
    ""

    Args:
        root (string): Root directory where the dataset should be saved.
        name:
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    node_type2id = {'pubchem_compound': 0,
                    'substructure': 1,
                    'gene': 2,
                    'kegg_pathway': 3,
                    'tissue': 4,
                    'omim_disease': 5,
                    'gene_family': 6,
                    'GO': 7,
                    'chebi': 8,
                    'sider': 9}

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['chem2bio2rdf.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        node2id, nid = {}, 0
        edge_type2id, etid = {}, 0

        srcs, dsts, edge_types, node_types = [], [], [], []
        with open(self.raw_paths[0], 'r', encoding='utf8') as f:
            for line in f.readlines():
                src, edge_type, dst = line.strip().split('\t')

                if edge_type not in edge_type2id:
                    edge_type2id[edge_type] = etid
                    etid += 1

                for i in [src, dst]:
                    if i not in node2id:
                        node2id[i] = nid
                        nid += 1

                        node_type1 = i.split("/")[-2]
                        node_type2 = i.split("/")[-3]
                        if node_type1 in self.node_type2id:
                            node_types.append(self.node_type2id[node_type1])
                        else:
                            node_types.append(self.node_type2id[node_type2])

                src = node2id[src]
                dst = node2id[dst]
                edge_type = edge_type2id[edge_type]

                srcs.append(src)
                dsts.append(dst)
                edge_types.append(edge_type)

        src = torch.tensor(srcs, dtype=torch.long)
        dst = torch.tensor(dsts, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        y = torch.tensor(node_types, dtype=torch.long)

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]

        num_edge_classes = len(edge_type2id)
        num_classes = len(self.node_type2id)
        data = Data(edge_index=edge_index, y=y,
                    num_nodes=num_nodes,
                    num_classes=num_classes, num_edge_classes=num_edge_classes,
                    edge_type=edge_type, edge_type2id=edge_type2id,
                    node2id=node2id, node_type2id=self.node_type2id)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}()'


if __name__ == '__main__':
    pass
