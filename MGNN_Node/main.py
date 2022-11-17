import os.path as osp
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from layer import *


def main(dataset_s, seed, verbose, num_epochs, auto_ml=False):
    print(f"running on seed: {seed}...")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1.load data
    transform = T.Compose([
        T.AddTrainValTestMask('train_rest', num_val=500, num_test=500)
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'NodeMGNN_DATA')
    dataset = Planetoid(path, dataset_s, transform=transform)
    data = dataset[0]
    x, edge_index = data.x, data.edge_index
    x = x.to(device)
    num_edge = edge_index.shape[-1]

    # 2.hyper-param
    num_filter = 1
    hidden_dim1 = 16  # 16
    if dataset_s == 'Cora':
        compress_dims = [6]  # 8
        att_act = torch.sigmoid
        layer_dropout = [0.5, 0.6]
        motif_dropout = 0.1
        att_dropout = 0.1
        mw_initializer = 'Xavier_Uniform'
        kernel_initializer = None
        bias_initializer = None
    elif dataset_s == 'CiteSeer':
        compress_dims = [6]
        att_act = torch.sigmoid
        layer_dropout = [0.5, 0.1]
        motif_dropout = 0.5
        att_dropout = 0.5
        mw_initializer = 'Kaiming_Uniform'
        kernel_initializer = 'Kaiming_Uniform'
        bias_initializer = None
    elif dataset_s == 'PubMed':
        compress_dims = [6, 6]
        att_act = torch.tanh
        layer_dropout = [0., 0.]
        motif_dropout = 0.3
        att_dropout = 0.
        mw_initializer = 'Xavier_Normal'
        kernel_initializer = 'Normal'
        bias_initializer = 'Zero'
        hidden_dim2 = dataset.num_classes
    else:
        raise Exception(f"Unknown dataset: {dataset_s}")

    # 3.preprocess
    row, col = edge_index.tolist()
    sp_mat = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))  # 不带自环
    edge_weight_norm = normalize_adj(sp_mat).data.reshape([-1, 1])
    mc = MotifCounter(dataset_s, [sp_mat], osp.join(path, dataset_s, 'processed'))
    motif_mats = mc.split_13motif_adjs()
    motif_mats = [convert_sparse_matrix_to_th_sparse_tensor(normalize_adj(motif_mat)).to(device) for motif_mat in
                  motif_mats]

    weight_index_data = np.array([range(num_filter)], dtype=np.int32).repeat(num_edge, axis=0)

    rel_type = [str(rel) for rel in set(weight_index_data.flatten().tolist())]
    graph_data = {('P', rel, 'P'): [[], []] for rel in rel_type}
    edge_data = {rel: [] for rel in rel_type}

    for rel in rel_type:
        for eid in range(weight_index_data.shape[0]):
            for j in range(num_filter):
                if str(weight_index_data[eid, j]) == rel:
                    graph_data[('P', rel, 'P')][0].append(row[eid])
                    graph_data[('P', rel, 'P')][1].append(col[eid])
                    edge_data[rel].append([edge_weight_norm[eid, 0]])

    graph_data = {rel: tuple(graph_data[rel]) for rel in graph_data}

    g = dgl.heterograph(graph_data).int().to(device)
    for rel in rel_type:
        g.edges[rel].data['edge_weight_norm'] = torch.tensor(edge_data[rel], dtype=torch.float32).to(device)

    # 5. build MGNN model
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = MotifConv(dataset.num_features, hidden_dim1, compress_dims[0], rel_type, dataset_s, motif_mats,
                                   mw_initializer, att_act, motif_dropout, att_dropout, aggr='sum')

            if len(compress_dims) > 1:
                self.conv2 = MotifConv(13 * compress_dims[0], hidden_dim2, compress_dims[1], rel_type, dataset_s,
                                       motif_mats,
                                       mw_initializer, att_act, motif_dropout, att_dropout, aggr='sum')
            else:
                self.register_parameter('conv2', None)

            self.dense = Linear(13 * compress_dims[-1], dataset.num_classes,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)

        def forward(self, g, h):
            h = F.dropout(h, p=layer_dropout[0], training=self.training)

            h = self.conv1(g, h)
            h = F.relu(h)

            h = F.dropout(h, p=layer_dropout[1], training=self.training)

            if self.conv2 is not None:
                h = self.conv2(g, h)
                h = F.relu(h)

            h = self.dense(h)

            return F.log_softmax(h, dim=1)

    model, data = Net().to(device), data.to(device)

    if dataset_s == 'Cora':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.011, weight_decay=16e-3)
    elif dataset_s == 'CiteSeer':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.011, weight_decay=20e-3)
    elif dataset_s == 'PubMed':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=2e-3)

    def train():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(g, x)[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

    @torch.no_grad()
    def evaluate():
        model.eval()
        log_probs, accs = model(g, x), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = log_probs[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    best_val_acc = test_acc = 0
    for epoch in range(1, num_epochs):
        train()
        train_acc, val_acc, tmp_test_acc = evaluate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if verbose:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))


if __name__ == '__main__':
    main('CiteSeer', seed=161, verbose=True, num_epochs=3001)
    # main('Cora', seed=55, verbose=True, num_epochs=3001)
    # main('PubMed', seed=99, verbose=True, num_epochs=3001)
