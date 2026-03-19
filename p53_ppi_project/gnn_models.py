from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise ModuleNotFoundError(
            "PyTorch is required for GCN training. Install torch in the project virtualenv first."
        ) from exc
    raise


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.linear(adj_norm @ x)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj_norm)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, alpha: float = 0.2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.attention = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        h = x @ self.weight
        num_nodes = h.size(0)
        
        h_i = h.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu(torch.matmul(a_input, self.attention).squeeze(-1))

        e = e.masked_fill(~adj_mask, float("-inf"))
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        return attention @ h


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, dropout=dropout)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, adj_mask)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_linear(x)
        return x
