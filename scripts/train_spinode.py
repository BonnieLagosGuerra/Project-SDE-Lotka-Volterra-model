import os
import numpy as np
import torch
from models.drift_net import DriftNet
from models.diffusion_net import DiffusionNet
from spinode.trainer import SPINODETrainer
from scripts.evaluate_spinode import evaluate_model
from scripts.simulate_data import simulate_trajectories

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    data_path = "data/moments.npz"

    # Generate data if not found
    if not os.path.exists(data_path):
        print("No data found. Simulating Lotka-Volterra trajectories...")
        from simulate_data import dt, N_steps, N_reps, T
        trajectories = simulate_trajectories()
        means = trajectories.mean(axis=0)
        covs = np.array([np.cov(trajectories[:, t, :].T) for t in range(trajectories.shape[1])])
        np.savez(data_path, time=np.linspace(0, T, N_steps+1), means=means, covs=covs, traj=trajectories)
        print(f"Dataset saved to {data_path}")
    else:
        print(f"Loading existing data from {data_path}...")

    data = np.load(data_path)
    time, means, covs = data["time"], data["means"], data["covs"]

    # Initialize models
    drift = DriftNet()
    diffusion = DiffusionNet()

    # Train
    print("\n=== Training SPINODE on Lotka-Volterra ===")
    trainer = SPINODETrainer(drift, diffusion, time, means, covs, device="cpu")
    drift_trained, diffusion_trained = trainer.train(n_epochs=200, batch_size=32, sequential=True)

    # Save models
    torch.save(drift_trained.state_dict(), "models/drift_trained.pt")
    torch.save(diffusion_trained.state_dict(), "models/diffusion_trained.pt")
    print("Models saved in models/")

    # Evaluate
    print("\n=== Evaluating Model Performance ===")
    evaluate_model()

if __name__ == "__main__":
    main()
