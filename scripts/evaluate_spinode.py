import numpy as np
import torch
import matplotlib.pyplot as plt
from models.drift_net import DriftNet
from models.diffusion_net import DiffusionNet
from spinode.trainer import SPINODETrainer
from scipy.stats import entropy, multivariate_normal

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def kl_divergence(mu_true, cov_true, mu_pred, cov_pred):
    """KL divergence between two Gaussian distributions."""
    n = mu_true.shape[0]
    cov_pred_inv = np.linalg.inv(cov_pred)
    term1 = np.trace(cov_pred_inv @ cov_true)
    diff = mu_pred - mu_true
    term2 = diff.T @ cov_pred_inv @ diff
    term3 = np.log(np.linalg.det(cov_pred) / np.linalg.det(cov_true))
    return 0.5 * (term1 + term2 - n + term3)

def evaluate_model():
    # Load trained models and data
    data = np.load("data/moments.npz")
    time, means, covs = data["time"], data["means"], data["covs"]

    drift = DriftNet()
    diffusion = DiffusionNet()
    drift.load_state_dict(torch.load("models/drift_trained.pt", map_location="cpu"))
    diffusion.load_state_dict(torch.load("models/diffusion_trained.pt", map_location="cpu"))

    trainer = SPINODETrainer(drift, diffusion, time, means, covs)
    pred_mean, pred_cov = trainer.predict()

    # Compute RMSE
    rmse_mean = rmse(pred_mean, means)
    rmse_cov = rmse(pred_cov, covs)
    print(f"RMSE (Mean): {rmse_mean:.4e}")
    print(f"RMSE (Covariance): {rmse_cov:.4e}")

    # Compute Validation Error (KL Divergence)
    kl_vals = []
    for i in range(0, len(time), max(1, len(time)//20)):
        kl = kl_divergence(means[i], covs[i], pred_mean[i], pred_cov[i])
        kl_vals.append(kl)
    print(f"Mean KL Divergence: {np.mean(kl_vals):.4e}")

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(time, means[:, 0], label="True μ₁")
    ax[0].plot(time, pred_mean[:, 0], '--', label="Pred μ₁")
    ax[0].plot(time, means[:, 1], label="True μ₂")
    ax[0].plot(time, pred_mean[:, 1], '--', label="Pred μ₂")
    ax[0].set_title("State Means vs Time")
    ax[0].legend()

    ax[1].plot(time, covs[:, 0, 0], label="True Σ₁₁")
    ax[1].plot(time, pred_cov[:, 0, 0], '--', label="Pred Σ₁₁")
    ax[1].plot(time, covs[:, 1, 1], label="True Σ₂₂")
    ax[1].plot(time, pred_cov[:, 1, 1], '--', label="Pred Σ₂₂")
    ax[1].set_title("Covariance Diagonal vs Time")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Optional: kernel density visualization
    try:
        from sklearn.neighbors import KernelDensity
        from scipy.stats import gaussian_kde
        traj = data["traj"]
        idxs = np.linspace(0, traj.shape[1]-1, 4, dtype=int)
        plt.figure(figsize=(10, 8))
        for j, idx in enumerate(idxs):
            plt.subplot(2, 2, j+1)
            kde = gaussian_kde(traj[:, idx, :].T)
            xgrid, ygrid = np.mgrid[0:2:100j, 0:2:100j]
            coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
            z = kde(coords).reshape(100, 100)
            plt.contourf(xgrid, ygrid, z, levels=30, cmap='viridis')
            plt.title(f"True Density at t={time[idx]:.2f}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Skipping KDE plot (requires scipy/sklearn).")

if __name__ == "__main__":
    evaluate_model()
