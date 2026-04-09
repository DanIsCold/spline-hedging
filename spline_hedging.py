import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import BSpline
from mpl_toolkits.mplot3d import Axes3D

# 1. SIMULATION PARAMETERS (Based on Mayenberger slide 11)
S0 = 1.0; K = 1.0; T = 0.5; sigma = 0.20; r = 0.0; cost = 0.005
n_paths = 5000; n_steps = 21; dt = T / n_steps
np.random.seed(42) # For reproducability

# 2. GENERATE GBM PATHS
Z = np.random.standard_normal((n_paths, n_steps))
S = np.zeros((n_paths, n_steps + 1))
S[:, 0] = S0
for t in range(n_steps):
    S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])
payoff = np.maximum(S[:, -1] - K, 0)

# 3. BASELINE: DISCRETE BSM DELTA HEDGING
def bsm_delta(S_t, t_elapsed):
    tau = T - t_elapsed
    if tau <= 0: return np.where(S_t > K, 1.0, 0.0)
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

bsm_hedges = np.zeros((n_paths, n_steps + 1))
t_grid = np.linspace(0, T, n_steps + 1)
for t in range(n_steps):
    bsm_hedges[:, t] = bsm_delta(S[:, t], t_grid[t])
bsm_hedges[: ,-1] = np.where(S[:, -1] > K, 1.0, 0.0)

# Calculate BSM P&L with transaction costs
bsm_trades = np.diff(bsm_hedges, prepend=0, axis=1)
bsm_tc = np.sum(np.abs(bsm_trades) * S * cost, axis=1)
bsm_pnl = payoff - np.sum(bsm_hedges[:, :-1] * np.diff(S, axis=1), axis=1) - bsm_tc

# 4. 1D SPLINE BASES
m_knots = np.concatenate(([0.5]*3, np.linspace(0.5, 1.5, 6), [1.5]*3))
t_knots = np.concatenate(([0.0]*3, np.linspace(0.0, T, 2), [T]*3))

def get_bspline_bases(x, knots, degree=3):
    n_bases = len(knots) - degree - 1
    bases = np.zeros((*x.shape, n_bases))
    for i in range(n_bases):
        c = np.zeros(n_bases); c[i] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        bases[..., i] = np.nan_to_num(spline(x)) # Replaces NaNs outside bounds with 0
    return bases

print("Pre-computing Spline Basis vectors...")
M = S[:, :-1] / K
Tau = T - np.linspace(0, T, n_steps + 1)[:-1]
Tau = np.broadcast_to(Tau, M.shape)

# Create the 12 features (8 for Moneyness, 4 for Time)
M_bases = get_bspline_bases(M, m_knots)
T_bases = get_bspline_bases(Tau, t_knots)
features = np.concatenate([M_bases, T_bases], axis=-1)

# Convert to PyTorch Tensors
features_pt = torch.tensor(features, dtype=torch.float32)
S_pt = torch.tensor(S, dtype=torch.float32)
payoff_pt = torch.tensor(payoff, dtype=torch.float32)

# 4. PYTORCH OPTIMISATION (Affine Function)
class SplineAffineHedge(nn.Module):
    def __init__(self):
        super().__init__()
        # 12 inputs (the bases) -> 1 output (the hedge) = 13 parameters :D
        self.linear = nn.Linear(12, 1)
    
    def forward(self, x):
        # Sigmoid ensures hedge ratio always stays betwen 0 and 1
        return torch.sigmoid(self.linear(x)).squeeze(-1)

model = SplineAffineHedge()
optimizer = optim.Adam(model.parameters(), lr=0.1)

print("Training Affine Spline Model...")
for epoch in range(300):
    optimizer.zero_grad()

    # Calculate hedges for all paths and time steps simultaneously
    hedges = model(features_pt)

    # Add final maturity hedge
    final_hedge = torch.where(S_pt[:, -1] > K, 1.0, 0.0).unsqueeze(-1)
    all_hedges = torch.cat([hedges, final_hedge], dim=1)

    # Calculate P&L and Transaction Costs
    trades = torch.diff(all_hedges, dim=1, prepend=torch.zeros((n_paths, 1)))
    tc = torch.sum(torch.abs(trades) * S_pt * cost, dim=1)

    dS = torch.diff(S_pt, dim=1)
    portfolio_pnl = torch.sum(hedges * dS, dim=1)
    pnl = payoff_pt - portfolio_pnl - tc

    # Objective: Minimize the Variance of the P&L
    loss = torch.var(pnl)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | P&L Variance: {loss.item():.6f}")

# 5. VISUALISATION
opt_spline_pnl = pnl.detach().numpy()

print(f"\nFinal BSM P&L Variance: {np.var(bsm_pnl):.6f}")
print(f"Final Spline P&L Variance: {np.var(opt_spline_pnl):.6f}")

plt.figure(figsize=(10, 5))
plt.hist(bsm_pnl, bins=50, alpha=0.5, label='Discrete BSM P&L', color='red')
plt.hist(opt_spline_pnl, bins=50, alpha=0.5, label='Spline Optimized P&L', color='blue')
plt.title('Replication P&L Distribution under Transaction Costs (c=0.5%)')
plt.xlabel('P&L')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pnl_distribution.png', dpi=300)
plt.show()

# 6. 3D SURFACE PLOT (Matches Slide 10)
# Create a grid of Moneyness and Time
m_grid = np.linspace(0.75, 1.25, 30)
t_grid = np.linspace(0.0, T, 30)
M_mesh, T_mesh = np.meshgrid(m_grid, t_grid)

# Flatten grids for prediction
M_flat = M_mesh.flatten()
T_flat = T_mesh.flatten()

# Get Spline Bases for the grid
M_bases_grid = get_bspline_bases(M_flat, m_knots)
T_bases_grid = get_bspline_bases(T_flat, t_knots)
features_grid = np.concatenate([M_bases_grid, T_bases_grid], axis=-1)
features_grid_pt = torch.tensor(features_grid, dtype=torch.float32)

# Predict hedges using trained model
with torch.no_grad():
    H_flat = model(features_grid_pt).numpy()

H_mesh = H_flat.reshape(M_mesh.shape)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(M_mesh, T_mesh, H_mesh, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('Optimized Affine Spline Hedge Surface', fontsize=14)
ax.set_xlabel('Moneyness (S/K)')
ax.set_ylabel('Time to Expiry (Years)')
ax.set_zlabel('Hedge Ratio (Delta)')
ax.view_init(elev=20, azim=-120)  # Adjust viewing angle
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Hedge Ratio')
plt.savefig('spline_surface_3d.png', dpi=300)
plt.show()

# Calculate BSM Delta on the same grid
def calculate_bsm_surface(M_grid, T_grid, sigma, r):
    tau = np.maximum(T_grid, 1e-6) 
    d1 = (np.log(M_grid) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta_surface = norm.cdf(d1)
    
    # At exactly T=0, Delta is a step function (Heaviside)
    delta_surface = np.where(T_grid <= 1e-7, np.where(M_grid > 1.0, 1.0, 0.0), delta_surface)
    return delta_surface

H_bsm_mesh = calculate_bsm_surface(M_mesh, T_mesh, sigma, r)

# Plotting BSM Surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(M_mesh, T_mesh, H_bsm_mesh, cmap='cividis', edgecolor='none', alpha=0.8)
ax.set_title('Theoretical BSM Delta Surface', fontsize=14)
ax.set_xlabel('Moneyness (S/K)')
ax.set_ylabel('Time to Expiry (Years)')
ax.set_zlabel('Delta')
ax.view_init(elev=20, azim=-120)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.savefig('bsm_surface_3d.png', dpi=300)
plt.show()
