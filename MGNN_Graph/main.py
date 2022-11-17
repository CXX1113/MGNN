import random
import os.path as osp
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from preprocess import *
from layer import MotifConv
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

# data_name = 'AIDS'  # 'MUTAG', 'PROTEINS' 'COLLAB'
seed = 55  # 55, 99999  (42: 0.809)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

compress_dims = [6, 6, 6]
att_act = torch.tanh
layer_dropout = [0.5, 0.5, 0.5]
motif_dropout = 0.3
att_dropout = 0.3
mw_initializer = 'Kaiming_Uniform'  # 'Xavier_Normal'
kernel_initializer = 'Orthogonal'
bias_initializer = 'Zero'

# hidden_dim1 = 88
epochs = 300
batch_size = 128  # 128 0.84   128*2 0.835
lr = 0.01
# lr_decay_factor = 0.5
# lr_decay_step_size = 50


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for graphs_data in loader:
        data = graphs_data[0]
        data = data.to(device)
        optimizer.zero_grad()
        out = model(graphs_data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for graphs_data in loader:
        data = graphs_data[0]
        data = data.to(device)
        with torch.no_grad():
            pred = model(graphs_data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for graphs_data in loader:
        data = graphs_data[0]
        data = data.to(device)
        with torch.no_grad():
            out = model(graphs_data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class Net(torch.nn.Module):
    def __init__(self, in_channel, hidden_dims, out_channel):
        super(Net, self).__init__()
        dim = 32*2
        hidden_dim1 = hidden_dims[0]
        hidden_dim2 = hidden_dims[1]
        self.conv1 = MotifConv(in_channel, hidden_dim1, compress_dims[0],
                               mw_initializer, att_act, motif_dropout, att_dropout, device)

        self.mlp1 = Sequential(Linear(13 * compress_dims[0], dim), BatchNorm1d(dim), ReLU(),
                               Linear(dim, dim), ReLU())

        self.conv2 = MotifConv(dim, hidden_dim2, compress_dims[1],
                               mw_initializer, att_act, motif_dropout, att_dropout, device)

        self.mlp2 = Sequential(Linear(13 * compress_dims[1], dim), BatchNorm1d(dim), ReLU(),
                               Linear(dim, dim), ReLU())

        self.conv3 = MotifConv(dim, hidden_dim2, compress_dims[2],
                               mw_initializer, att_act, motif_dropout, att_dropout, device)

        self.mlp3 = Sequential(Linear(13 * compress_dims[2], dim), BatchNorm1d(dim), ReLU(),
                               Linear(dim, dim), ReLU())

        # self.dense = MyLinear(dim, dataset.num_classes,
        #                       kernel_initializer=kernel_initializer,
        #                       bias_initializer=bias_initializer)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channel)

    def forward(self, graphs_data):
        graph_org = graphs_data[0]
        h = graph_org.x.to(device)
        batch = graph_org.batch
        h = F.dropout(h, p=layer_dropout[0], training=self.training)

        h = self.conv1(graphs_data, h)
        # h = F.relu(h)
        h = self.mlp1(h)

        h = F.dropout(h, p=layer_dropout[1], training=self.training)

        h = self.conv2(graphs_data, h)
        # h = F.relu(h)
        h = self.mlp2(h)

        h = F.dropout(h, p=layer_dropout[2], training=self.training)

        h = self.conv3(graphs_data, h)
        # h = F.relu(h)
        h = self.mlp3(h)

        # h = self.dense(h)
        h = global_add_pool(h, batch)

        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)


def cross_validation_with_val_set(data_name, folds, weight_decay, logger=None, gen_x='max_degree'):
    print(f"using dataset: {data_name} with seed={seed}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GraphMGNN_DATA')
    dataset = TUDataset(path, name=data_name, pre_transform=ScaledNormalize(), use_node_attr=True)

    if dataset.data.x is None:
        if gen_x == 'max_degree':
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
        else:
            dataset.transform = T.LocalDegreeProfile()

    print("num_feature:", dataset.num_features)

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        t_start = time.perf_counter()

        train_datasets = []
        val_datasets = []
        test_datasets = []

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

        # the batch of 13 motif adjs
        for i in range(1, 14):
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GraphMGNN_DATA', f'M{i}')
            dataset_motif = TUDataset(path, name=data_name, pre_transform=MotifGraph(f'M{i}', data_name))
            train_dataset = dataset_motif[train_idx]
            test_dataset = dataset_motif[test_idx]
            val_dataset = dataset_motif[val_idx]
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            val_datasets.append(val_dataset)

        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size, shuffle=True)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size, shuffle=False)
        test_loader = DataLoader(ConcatDataset(test_datasets), batch_size, shuffle=False)

        model = Net(dataset.num_features, [16, dataset.num_classes], dataset.num_classes).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-3)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            print(f"fold={fold} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_losses[-1]:.4f} test_acc={accs[-1]:.4f}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        print(f"[end] {fold + 1}-th fold training finished(cost{t_end - t_start:.2f}s).")

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)

    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


if __name__ == '__main__':
    # cross_validation_with_val_set('AIDS', folds=5, weight_decay=0, gen_x='max_degree')  # 0.994 ± 0.005
    # cross_validation_with_val_set('ENZYMES', folds=5, weight_decay=0, gen_x='max_degree')  # 0.310 ± 0.078
    cross_validation_with_val_set('MUTAG', folds=5, weight_decay=0, gen_x='max_degree')  # 0.835 ± 0.057


