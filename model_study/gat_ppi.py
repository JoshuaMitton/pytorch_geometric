import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from sklearn import metrics

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='test')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
dataset_name = 'PPI'


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(loaders):
    model.eval()

    accs = []
    for loader in loaders:
        total_micro_f1 = 0
        for data in loader:
            with torch.no_grad():
                out = model(data.x.to(device), data.edge_index.to(device))
            pred = (out > 0).float().cpu()
            micro_f1 = metrics.f1_score(data.y, pred, average='micro')
            total_micro_f1 += micro_f1 * data.num_graphs
        accs.append(total_micro_f1 / len(loader.dataset))
    return accs


history = {'train_acc':[], 'val_acc':[], 'test_acc':[]}
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test([train_loader, val_loader, test_loader])
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['test_acc'].append(test_acc)
    print('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, loss, train_acc, val_acc, test_acc))

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

