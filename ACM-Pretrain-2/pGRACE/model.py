from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pGRACE import contrastive_loss


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.cluster_temperature = 0.7
        self.cluster_num = 3
        self.loss_device = torch.device("cuda")
        self.criterion_cluster = contrastive_loss.ClusterLoss(self.cluster_num, self.cluster_temperature,
                                                              self.loss_device).to(self.loss_device)
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

        self.instance_projector = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_proj_hidden),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x1: torch.Tensor, edge_index1: torch.Tensor, x2: torch.Tensor,
                edge_index2: torch.Tensor) -> torch.Tensor:
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)

        z_i = F.normalize(self.instance_projector(z1), dim=1)
        z_j = F.normalize(self.instance_projector(z2), dim=1)

        c_i = self.cluster_projector(z1)
        c_j = self.cluster_projector(z2)

        return z_i, z_j, c_i, c_j

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z_i: torch.Tensor, z_j: torch.Tensor, c_i: torch.Tensor, c_j: torch.Tensor, mean: bool = True,
             batch_size: Optional[int] = None):
        h1 = z_i
        h2 = z_j

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        loss_instance = ret.mean() if mean else ret.sum()

        loss_cluster = self.criterion_cluster(c_i, c_j)

        loss = loss_instance + loss_cluster
        return loss

    # Obtain Clustering results
    def forward_cluster(self, x1: torch.Tensor, edge_index1: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x1, edge_index1)
        c = self.cluster_projector(z)
        c = torch.argmax(c, dim=1)
        return c