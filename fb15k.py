# This code is generated from `fb15k.ipynb` for debugging purpose only.
# %% [markdown]
# # Demo for training TransE model on the FB15k_237 dataset
#
# Code is based on [PyTorch Geometric Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/kge_fb15k_237.py)

# %%
from torch_geometric.nn.kge import KGEModel
from torch_geometric.nn.kge.loader import KGTripletLoader
import torch.optim as optim
import torch
from torch_geometric.nn import TransE
from torch_geometric.datasets import FB15k_237
from torch_geometric.data import Dataset

# %%
data_train = FB15k_237('./data/fb15k', split='train')[0]
data_val = FB15k_237('./data/fb15k', split='val')[0]
data_test = FB15k_237('./data/fb15k', split='test')[0]

# %%
print(f'# of graph:    {len(data_train)}')
print(f'# of nodes:    {data_train.num_nodes}')
print(f'# of edges:    {data_train.num_edges}')
print(f'# of node features: {data_train.num_node_features}')
print(f'# of edge features:    {data_train.num_edge_features}')
print(f'# of edge types: {data_train.num_edge_types}')

# %%
num_nodes = data_train.num_nodes
num_relations = data_train.num_edge_types
hidden_channels = 50

# %%

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
model = TransE(
    num_nodes=num_nodes,
    num_relations=num_relations,
    hidden_channels=hidden_channels,
)

# %%
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %%

# %%
loader_train = KGTripletLoader(
    head_index=data_train.edge_index[0],
    rel_type=data_train.edge_type,
    tail_index=data_train.edge_index[1],
    batch_size=2000,
    shuffle=True,
)


# %%
def train(model: KGEModel, loader: KGTripletLoader):
    total_loss = 0
    total_examples = 0

    model.train()
    for triple in loader:
        head_index, rel_type, tail_index = triple
        head_index = head_index.to(device)
        rel_type = rel_type.to(device)
        tail_index = tail_index.to(device)

        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()

    return total_loss / total_examples

# %%


@torch.no_grad()
def test(model: KGEModel, dataset: Dataset):
    model.eval()
    dataset = dataset.to(device)
    mean_rank, hits_at_k = model.test(
        head_index=dataset.edge_index[0],
        rel_type=dataset.edge_type,
        tail_index=dataset.edge_index[1],
        batch_size=10000,
        k=10,
    )
    return mean_rank, hits_at_k


# %%
model = model.to(device)
for epoch in range(1, 501):
    loss = train(model, loader_train)
    if epoch % 10 == 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, hits = test(model, data_val)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val Hits@10: {hits:.4f}')

rank, hits_at_10 = test(model, data_test)
print(f'Test Mean Rank: {rank:.2f}, Test Hits@10: {hits_at_10:.4f}')

# %%
