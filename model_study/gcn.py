import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['Cora','CiteSeer','PubMed', 'NELL'], default='Cora')
args = parser.parse_args()

dataset_name = args.dataset
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        self.conv1 = ChebConv(data.num_features, 16, K=2)
        self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


history = {'train_acc':[], 'val_acc':[], 'test_acc':[]}
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    #history['test_acc'].append(test_acc)
    history['test_acc'].append(tmp_test_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

plt.figure()
plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.plot(history['test_acc'])
plt.ylim([0,1])
plt.title(('Model Accuracy - Testing:'+str(history['test_acc'][-1])))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
plt.savefig(('gcn/'+dataset_name+'_Training.png'))
plt.show()

