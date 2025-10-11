import torch
import torch.nn as nn

class DiffusionNet(nn.Module):
    """
    Neural network approximating the diffusion term g2(x; θ2)
    of the stochastic Lotka-Volterra competition model.
    Input: [x1, x2]
    Output: [g2_1, g2_2]  (positive)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus()  # ensures non-negative diffusion
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = DiffusionNet()
    x = torch.randn(5, 2)
    print("Output (positive):", model(x))
