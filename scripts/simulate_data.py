import numpy as np
import os

# Parameters
k1, k2 = 0.4, 0.5
dt = 0.01
T = 5.0
N_steps = int(T / dt)
N_reps = 1000  # number of stochastic trajectories
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Coexistence equilibrium
x_eq1 = (1 - k1) / (1 - k1 * k2)
x_eq2 = (1 - k2) / (1 - k1 * k2)

# Drift and diffusion
def g1(x):
    x1, x2 = x[..., 0], x[..., 1]
    return np.stack([
        x1 * (1 - x1 - k1 * x2),
        x2 * (1 - x2 - k2 * x1)
    ], axis=-1)

def g2(x):
    x1, x2 = x[..., 0], x[..., 1]
    return np.stack([
        x1 * (x2 - x_eq2),
        x2 * (x1 - x_eq1)
    ], axis=-1)

# Simulation (Euler–Maruyama)
def simulate_trajectories(N=N_reps, steps=N_steps, dt=dt):
    x = np.zeros((N, 2))
    x[:, 0] = np.random.uniform(0.3, 1.2, size=N)
    x[:, 1] = np.random.uniform(0.3, 1.2, size=N)
    traj = np.zeros((N, steps+1, 2))
    traj[:, 0, :] = x

    for t in range(steps):
        drift = g1(x)
        diff = np.sqrt(2 * np.maximum(g2(x), 0))
        noise = np.random.randn(N, 2)
        x = x + drift * dt + diff * np.sqrt(dt) * noise
        x = np.clip(x, 0, None)  # prevent negative values
        traj[:, t+1, :] = x
    return traj

print("Simulating trajectories...")
trajectories = simulate_trajectories()

# Compute mean and covariance over time
means = trajectories.mean(axis=0)  # (steps+1, 2)
covs = np.array([
    np.cov(trajectories[:, t, :].T) for t in range(trajectories.shape[1])
])  # (steps+1, 2, 2)

time = np.linspace(0, T, N_steps+1)

np.savez(
    os.path.join(save_dir, "moments.npz"),
    time=time, means=means, covs=covs, traj=trajectories
)
print(f"Saved to {save_dir}/moments.npz")
