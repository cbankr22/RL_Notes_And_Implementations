import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
from torch_geometric.datasets import Planetoid

class GNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNLayer, self).__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim)

    def forward(self, node_features, adj_matrix):
        # node_features is matrix of size num_nodes x input_dim
        # node_features = [1 1], each row i is the feature vector of the node i
        #                 [0 1]
        # adj_matrix is a matrix of size num_nodes x num_nodes
        # adj_matrix = [0 1], for a connected graph with 2 nodes # adj_matrix is always symmetric, something here
        #              [1 0]
        sum_neighbors = torch.mm(adj_matrix, node_features)

        # sum_neighbors = [0 1][1 1] = [0 1]
        #                 [1 0][0 1]   [1 1]
        concat_features = torch.cat([node_features, sum_neighbors], dim=1)

        # concat_features = [1 1]
        #                   [0 1]
        #                   [0 1]
        #                   [1 1]
        return F.relu(self.fc(concat_features))


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.layer1 = GNLayer(input_dim, hidden_dim)
        self.layer2 = GNLayer(hidden_dim, output_dim)

    def forward(self, node_features, adj_matrix):
        h = self.layer1(node_features, adj_matrix)
        h = self.layer2(h, adj_matrix)
        return h

if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    node_features = data.x
    adj_matrix = torch.eye(data.num_nodes)

    for edge in data.edge_index.t():
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    labels = data.y
    train_mask = data.train_mask
    test_mask = data.train_mask

    input_dim = node_features.shape[1]
    hidden_dim = 64
    output_dim = dataset.num_classes
    learning_rate = 0.0001
    epochs = 200

    model = GNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(node_features, adj_matrix)

        # Compute the loss
        loss = criterion(out[train_mask], labels[train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    out = model(node_features, adj_matrix)
    _, pred = out.max(dim=1)
    correct = float(pred[test_mask].eq(labels[test_mask]).sum().item())
    accuracy = correct / test_mask.sum().item()

    print(f"Test Accuracy: {accuracy:.4f}")