# Project-SDE-Lotka-Volterra-model
A project that attempts to create a model capable of solving the Lotka–Volterra system using SDEs.

## 🧠 How to Run This Project

### 🚀 Quick Start

#### Prerequisites

- **Python 3.11+**  
- **pip** or **conda** for package management  

---

### ⚙️ Local Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository>
   cd Project-SDE-Lotka-Volterra-model
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   # ✅ Option 1 — recommended (run from the project root)
   python -m scripts/train_spinode.py
   # ⚙️ Option 2 — run directly
   python scripts/train_spinode.py
   ```
---

## 🧩 About the Model

This implementation is inspired by the paper **"Stochastic Physics-Informed Neural Ordinary Differential Equations"** by Jared O'Leary, Joel A. Paulson, and Ali Mesbah (University of California, Berkeley; The Ohio State University).

The SPINODE framework extends the concept of Physics-Informed Neural Networks (PINNs) to stochastic differential equations (SDEs), combining Neural Ordinary Differential Equations (Neural ODEs) with the learning of hidden physical dynamics in noisy systems.

SPINODE uses uncertainty propagation through the Unscented Transform (UT) and a moment-matching loss to train neural networks that approximate both drift and diffusion terms in stochastic systems.

In this project, we apply the SPINODE framework to a competitive Lotka–Volterra system with a coexistence equilibrium, governed by:

$$
\begin{aligned}
dx_1 &= g_1(x_1, x_2)dt + \sqrt{2g_2(x_1, x_2)}dw_1, \\
dx_2 &= g_1(x_2, x_1)dt + \sqrt{2g_2(x_2, x_1)}dw_2,
\end{aligned}
$$

where $g_1$ and $g_2$ represent the hidden physics of the system — the drift and diffusion coefficients — learned directly from stochastic trajectory data.

Two separate neural networks are trained:

- **One for the drift term** $g_1(x_1, x_2)$  
- **One for the diffusion term** $g_2(x_1, x_2)$

Each network takes the state variables $(x_1, x_2)$ as inputs and outputs the corresponding components of $g_1$ or $g_2$.

### 🔬 References and Inspiration

This project draws upon and integrates ideas from the following works:

- **O'Leary, J., Paulson, J.A., & Mesbah, A. (2023).**  
  *Stochastic Physics-Informed Neural Ordinary Differential Equations.*  
  University of California, Berkeley / The Ohio State University.

- **El Janati Elidrissi, Y., & Efstathiadis, G. (2023).**  
  *PINN-Based SDE Solver.*  
  Harvard T.H. Chan School of Public Health.

- **Olguín, D. (2024).**  
  *The Math Behind the Magic: Neural Networks, Theory and Practice.*  
  Encuentro Nacional de Ingeniería Matemática 2024,  
  with J. Fontbona, J. Maass, and C. Muñoz.

### 🧾 Technical Notes

The goal of this project is to demonstrate how neural networks can learn nonlinear stochastic dynamics without assuming an explicit analytic form for the governing equations.

By combining SPINODE with the Lotka–Volterra system, this case study showcases how data-driven neural differential equation methods can infer the underlying physical structure of complex interacting systems.

### 📂 Repository Structure
   ```bash
   Project-SDE-Lotka-Volterra-model/
   │
   ├── scripts/
   │   ├── train_spinode.py        # SPINODE model training
   │   ├── utils.py                # Helper functions (metrics, propagation, etc.)
   │   └── plot_results.py         # Visualization of trajectories and learned dynamics
   │
   ├── data/
   │   ├── trajectories/           # Stochastic trajectory datasets
   │   └── results/                # Training results
   │
   ├── models/
   │   ├── drift_net.py            # Neural network for drift term
   │   ├── diffusion_net.py        # Neural network for diffusion term
   │   └── checkpoint/             # Saved model weights
   │
   ├── requirements.txt
   └── README.md
   ```


## 🧩 Author

Academic project inspired by research on SDEs, PINNs, and Neural ODEs, developed for mathematical and computational exploration.

