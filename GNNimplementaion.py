import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Step 1: Create a sample graph dataset
# Nodes represent transactions; edges represent relationships
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Edges between nodes
x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)  # Node features (e.g., transaction metadata)
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)  # Labels: 0 = legitimate, 1 = fraudulent

data = Data(x=x, edge_index=edge_index, y=y)

# Step 2: Define the Graph Neural Network model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)  # First GCN layer
        self.conv2 = GCNConv(16, 2)  # Second GCN layer (output: two classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 3: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Step 4: Evaluate the model
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == data.y).sum()
accuracy = int(correct) / len(data.y)
print(f'Accuracy: {accuracy:.4f}')
