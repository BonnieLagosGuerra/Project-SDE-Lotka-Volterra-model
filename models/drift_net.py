import torch
import torch.nn as nn

class DriftNet(nn.Module):
    """
    Neural network approximating the drift term g1(x; θ1)
    of the stochastic Lotka-Volterra competition model.
    Input: [x1, x2]
    Output: [g1_1, g1_2]
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = DriftNet()
    x = torch.randn(5, 2)
    print("Output shape:", model(x).shape)