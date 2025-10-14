import torch
import torch.nn as nn
import torch.linalg as LA
from scipy.linalg import sqrtm

class SPINODE_ODE(nn.Module):
    """
    Defines deterministic ODEs for mean and covariance evolution
    using Unscented Transform (UT-2M) from SPINODE paper.
    """
    def __init__(self, drift_net, diffusion_net, alpha=1e-3, beta=2.0, kappa=0.0):
        super().__init__()
        self.drift = drift_net
        self.diffusion = diffusion_net
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _unscented_transform(self, mu, cov):
        """Generate sigma points and weights (2n + 1)."""
        n = mu.shape[-1]
        lam = self.alpha**2 * (n + self.kappa) - n
        # U = LA.cholesky((n + lam) * cov)
        U = sqrtm((n + lam) * cov)
        sigma_pts = [mu]
        for i in range(n):
            sigma_pts.append(mu + U[:, i])
        for i in range(n,2*n):
            sigma_pts.append(mu - U[:, i])
        sigma_pts = torch.stack(sigma_pts)

        # Weights
        Wm = torch.ones(2 * n + 1) / (2 * (n + lam))
        Wc = Wm.clone()
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / ((n + lam) - (1 - self.alpha**2 + self.beta))
        return sigma_pts, Wm.to(mu.device), Wc.to(mu.device)

    def forward(self, t, y):
        """
        ODE system in vectorized form.
        y = [μx, Σx_flat]
        Returns dy/dt
        """
        n = 2  # Lotka–Volterra has 2 states
        mu = y[:n]
        cov_flat = y[n:]
        cov = cov_flat.reshape(n, n)

        sigma_pts, Wm, Wc = self._unscented_transform(mu, cov)
        F_vals = self.drift(sigma_pts)

        mu_dot = torch.sum(Wm[:, None] * F_vals, dim=0)
        F_mean = mu_dot.detach()

        # Compute covariance derivative
        diff_terms = []
        for i in range(sigma_pts.size(0)):
            d = (F_vals[i] - F_mean).unsqueeze(1)
            diff_terms.append(Wc[i] * (d @ d.T))
        cov_dot = sum(diff_terms)

        # Add diffusion effect
        g2_vals = self.diffusion(sigma_pts)
        G2_mean = torch.sum(Wm[:, None] * g2_vals, dim=0)
        cov_dot += torch.diag(G2_mean)

        dydt = torch.cat([mu_dot, cov_dot.flatten()])
        return dydt
