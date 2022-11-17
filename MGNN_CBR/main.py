import random
import torch_geometric.transforms as T
from layer import *


def main(dataset_s, seed, verbose, num_epochs, auto_ml=False):
    print(f"running on seed: {seed}...")
    # dataset_s = "Cora"  # "Cora", "CiteSeer", "PubMed"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 1.load data
    transform = T.Compose([
        T.AddTrainValTestMask('train_rest', num_val=5000, num_test=5000),
        T.LocalDegreeProfile()
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'NodeMGNN_DATA')
    dataset = Chem2Bio2RDF(path, dataset_s, pre_transform=transform)
    data = dataset[0]
    x, edge_index = data.x, data.edge_index
    x = x.to(device)
    input_dim = x.shape[-1]
    print("input_dim: ", input_dim)
    print("num_class: ", dataset.num_classes)
    num_edge = edge_index.shape[-1]

    # 2.hyper param
    hidden_dim1 = 16
    if dataset_s == 'chem2bio2rdf':
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
    sp_mat = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))
    mc = MotifCounter(dataset_s, [sp_mat], osp.join(path, dataset_s, 'processed'))
    sp_eye = sp.eye(x.shape[0], dtype=np.int32)
    sp_mat = sp_mat + sp_mat.transpose()
    sp_mat = sp_mat + sp_eye
    sp_mat_norm = normalize_adj(sp_mat).astype(dtype=np.float32)
    motif_mats = mc.split_13motif_adjs()
    motif_mats = [convert_sparse_matrix_to_th_sparse_tensor(normalize_adj(motif_mat)).to(device) for motif_mat in
                  motif_mats]

    g = dgl.from_scipy(sp_mat_norm, eweight_name='edge_weight_norm').to(device)

    # 4. build MGNN model
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = MotifConv(dataset.num_features, hidden_dim1, compress_dims[0], dataset_s, motif_mats,
                                   mw_initializer, att_act, motif_dropout, att_dropout, aggr='sum')

            if len(compress_dims) > 1:
                self.conv2 = MotifConv(13 * compress_dims[0], hidden_dim2, compress_dims[1], dataset_s,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.011, weight_decay=5e-3)

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
    main('chem2bio2rdf', seed=186, verbose=True, num_epochs=5001)

