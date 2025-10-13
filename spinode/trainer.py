import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import numpy as np
from spinode.ode_system import SPINODE_ODE

class SPINODETrainer:
    def __init__(self, drift_net, diffusion_net, time, means, covs, device="cpu"):
        self.device = device
        self.time = torch.tensor(time, dtype=torch.float32, device=device)
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.covs = torch.tensor(covs, dtype=torch.float32, device=device)
        self.drift_net = drift_net.to(device)
        self.diffusion_net = diffusion_net.to(device)
        self.system = SPINODE_ODE(drift_net, diffusion_net).to(device)

    def moment_loss(self, pred, true_mean, true_cov):
        n = 2
        pred_mean = pred[:, :n]
        pred_cov = pred[:, n:].reshape(-1, n, n)
        loss_mean = torch.mean((pred_mean - true_mean) ** 2)
        loss_cov = torch.mean((pred_cov - true_cov) ** 2)
        return loss_mean + loss_cov

    def train(
        self, lr_drift=1e-3, lr_diff=1e-3, n_epochs=300,
        batch_size=32, sequential=True
    ):
        """Train drift and diffusion networks sequentially."""
        optimizer_drift = optim.Adam(self.drift_net.parameters(), lr=lr_drift)
        optimizer_diff = optim.Adam(self.diffusion_net.parameters(), lr=lr_diff)

        # Train drift first
        if sequential:
            print("\nTraining DriftNet (θ₁)...")
            for epoch in tqdm(range(n_epochs)):
                optimizer_drift.zero_grad()
                y0 = torch.cat([
                    self.means[0],
                    self.covs[0].flatten()
                ])
                pred = odeint(
                    self.system, y0, self.time,
                    method="dopri5", atol=1e-5, rtol=1e-4
                )
                loss = self.moment_loss(pred, self.means, self.covs)
                loss.backward()
                optimizer_drift.step()
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}, Drift Loss: {loss.item():.4e}")

        # Train diffusion next
        print("\nTraining DiffusionNet (θ₂)...")
        for epoch in tqdm(range(n_epochs)):
            optimizer_diff.zero_grad()
            y0 = torch.cat([
                self.means[0],
                self.covs[0].flatten()
            ])
            pred = odeint(
                self.system, y0, self.time,
                method="dopri5", atol=1e-5, rtol=1e-4
            )
            loss = self.moment_loss(pred, self.means, self.covs)
            loss.backward()
            optimizer_diff.step()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Total Loss: {loss.item():.4e}")

        print("\nTraining complete")
        return self.drift_net, self.diffusion_net

    @torch.no_grad()
    def predict(self):
        """Generate predicted mean and covariance trajectories."""
        y0 = torch.cat([
            self.means[0],
            self.covs[0].flatten()
        ])
        pred = odeint(
            self.system, y0, self.time,
            method="dopri5", atol=1e-5, rtol=1e-4
        )
        n = 2
        pred_mean = pred[:, :n].cpu().numpy()
        pred_cov = pred[:, n:].reshape(-1, n, n).cpu().numpy()
        return pred_mean, pred_cov
