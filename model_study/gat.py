import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['Cora','CiteSeer','PubMed', 'PPI'], default='Cora')
args = parser.parse_args()

dataset_name = args.dataset
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


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
for epoch in range(1, 201):
    train()
    train_acc, val_acc, test_acc = test()
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['test_acc'].append(test_acc)
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, val_acc, test_acc))

plt.figure()
plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.plot(history['test_acc'])
plt.ylim([0,1])
plt.title(('Model Accuracy - Testing:'+str(history['test_acc'][-1])))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
plt.savefig(('gat/'+dataset_name+'_Training.png'))
plt.show()

