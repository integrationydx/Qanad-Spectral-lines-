"""
Spectral Error Indicators for Neural PDE Solvers
Phase 4: Allen-Cahn Benchmark (Real PINN Dynamics)

- Solves the Allen-Cahn equation using a real PyTorch PINN.
- Captures true intermediate gradient convergence snapshots.
- Applies DMD to extract spectral features.
- Trains a CNN to predict true local error.
- Compares correlation against the baseline PDE Residual.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("PHASE 4: Allen-Cahn Benchmark (Real PINN Dynamics)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. GROUND TRUTH: Numerical Allen-Cahn Solver
# ─────────────────────────────────────────────────────────────
Nx = 256
Nt_pred = 100
x = np.linspace(-1, 1, Nx)
t_eval = np.linspace(0, 1, Nt_pred)
dx = x[1] - x[0]

def allen_cahn_rhs(t, u):
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
    u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
    return 0.0001 * u_xx - 5 * u**3 + 5 * u

print("Computing exact numerical solution for Allen-Cahn...")
u0 = (x**2) * np.cos(np.pi * x)
sol = solve_ivp(allen_cahn_rhs, [0, 1], u0, t_eval=t_eval, method='RK45')
U_exact = sol.y  # [Nx, Nt]

# ─────────────────────────────────────────────────────────────
# 2. REAL PINN ARCHITECTURE & TRAINING
# ─────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

print("Training PyTorch PINN (collecting real snapshots)...")
pinn = PINN()
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

# Collocation points
N_f = 8000
x_f = np.random.uniform(-1, 1, (N_f, 1))
t_f = np.random.uniform(0, 1, (N_f, 1))
x_f_t = torch.tensor(x_f, dtype=torch.float32, requires_grad=True)
t_f_t = torch.tensor(t_f, dtype=torch.float32, requires_grad=True)

# Initial conditions
N_ic = 200
x_ic = np.random.uniform(-1, 1, (N_ic, 1))
t_ic = np.zeros((N_ic, 1))
u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
x_ic_t = torch.tensor(x_ic, dtype=torch.float32)
t_ic_t = torch.tensor(t_ic, dtype=torch.float32)
u_ic_t = torch.tensor(u_ic, dtype=torch.float32)

# Boundary conditions (Periodic)
N_bc = 200
t_bc = np.random.uniform(0, 1, (N_bc, 1))
x_bc_left = np.full((N_bc, 1), -1.0)
x_bc_right = np.full((N_bc, 1), 1.0)

t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True)
x_left_t = torch.tensor(x_bc_left, dtype=torch.float32, requires_grad=True)
x_right_t = torch.tensor(x_bc_right, dtype=torch.float32, requires_grad=True)

# Grid for snapshots
X_grid, T_grid = np.meshgrid(x, t_eval, indexing='ij')
x_grid_flat = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32)
t_grid_flat = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32)

snapshots = []
epochs = 2000
snapshot_interval = 40

for ep in range(epochs):
    optimizer.zero_grad()
    
    # Physics Loss
    u_f = pinn(x_f_t, t_f_t)
    u_t = torch.autograd.grad(u_f, t_f_t, torch.ones_like(u_f), create_graph=True)[0]
    u_x = torch.autograd.grad(u_f, x_f_t, torch.ones_like(u_f), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f_t, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t - 0.0001 * u_xx + 5 * u_f**3 - 5 * u_f
    loss_f = torch.mean(f**2)
    
    # IC Loss
    u_pred_ic = pinn(x_ic_t, t_ic_t)
    loss_ic = torch.mean((u_pred_ic - u_ic_t)**2)
    
    # BC Loss
    u_left = pinn(x_left_t, t_bc_t)
    u_right = pinn(x_right_t, t_bc_t)
    u_x_left = torch.autograd.grad(u_left, x_left_t, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_right_t, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc = torch.mean((u_left - u_right)**2) + torch.mean((u_x_left - u_x_right)**2)
    
    loss = loss_f + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    optimizer.step()
    
    if (ep + 1) % snapshot_interval == 0:
        with torch.no_grad():
            u_snap = pinn(x_grid_flat, t_grid_flat).view(Nx, Nt_pred).detach().numpy()
            snapshots.append(u_snap)

print(f"Collected {len(snapshots)} real PINN snapshots.")

# Final Prediction and Residual
x_grid_t = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)
t_grid_t = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)
u_pred_final = pinn(x_grid_t, t_grid_t)
u_t_final = torch.autograd.grad(u_pred_final, t_grid_t, torch.ones_like(u_pred_final), create_graph=True)[0]
u_x_final = torch.autograd.grad(u_pred_final, x_grid_t, torch.ones_like(u_pred_final), create_graph=True)[0]
u_xx_final = torch.autograd.grad(u_x_final, x_grid_t, torch.ones_like(u_x_final), create_graph=True)[0]
f_final = (u_t_final - 0.0001 * u_xx_final + 5 * u_pred_final**3 - 5 * u_pred_final).detach().numpy().reshape(Nx, Nt_pred)
final_pinn = u_pred_final.detach().numpy().reshape(Nx, Nt_pred)

true_error_spatial = np.mean(np.abs(final_pinn - U_exact), axis=1)
residual_spatial = np.mean(np.abs(f_final), axis=1)

# ─────────────────────────────────────────────────────────────
# 3. DMD PIPELINE
# ─────────────────────────────────────────────────────────────
S = np.array([s.flatten() for s in snapshots]).T
X_mat = S[:, :-1]
Xp_mat = S[:, 1:]

U, sigma, Vt = svd(X_mat, full_matrices=False)
r = 25
U_r = U[:, :r]
sigma_r = sigma[:r]
Vt_r = Vt[:r, :]

A_tilde = U_r.T @ Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r)
eigvals, W = np.linalg.eig(A_tilde)
Phi = Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r) @ W

energies = np.abs(eigvals) * np.linalg.norm(Phi, axis=0)
energies = energies / (energies.sum() + 1e-12)
H = -np.sum(energies * np.log(energies + 1e-12))

Phi_spatial = np.abs(Phi).reshape(Nx, Nt_pred, r).mean(axis=1)
feat_eig = np.tile(np.abs(eigvals[:r]), (Nx, 1))
feat_ent = np.full((Nx, 1), H)
feat_energy = np.tile(energies[:r], (Nx, 1))
X_features = np.hstack([Phi_spatial, feat_eig, feat_ent, feat_energy])

# ─────────────────────────────────────────────────────────────
# 4. 1D CNN ERROR PREDICTOR (5-FOLD OOF)
# ─────────────────────────────────────────────────────────────
X_cnn_pad = np.pad(Phi_spatial, ((10, 10), (0, 0)), mode='edge')
windows = np.array([X_cnn_pad[i:i+21].T for i in range(Nx)]) # [Nx, r, 21]

class Conv1DNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 21, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.conv(x)

pred_cnn = np.zeros(Nx)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(windows):
    win_tr, win_va = windows[train_idx], windows[val_idx]
    y_tr, y_va = true_error_spatial[train_idx], true_error_spatial[val_idx]
    
    train_mean = win_tr.mean(axis=0, keepdims=True)
    train_std = win_tr.std(axis=0, keepdims=True) + 1e-8
    win_tr = (win_tr - train_mean) / train_std
    win_va = (win_va - train_mean) / train_std
    
    y_max = y_tr.max() + 1e-8
    y_tr_scaled = y_tr / y_max
    
    win_tr_t = torch.tensor(win_tr, dtype=torch.float32)
    win_va_t = torch.tensor(win_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_scaled, dtype=torch.float32).unsqueeze(1)
    
    loader = DataLoader(TensorDataset(win_tr_t, y_tr_t), batch_size=32, shuffle=True)
    cnn = Conv1DNet(in_ch=r)
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.005)
    
    cnn.train()
    for _ in range(80):
        for bx, by in loader:
            optimizer_cnn.zero_grad()
            loss_c = nn.MSELoss()(cnn(bx), by)
            loss_c.backward()
            optimizer_cnn.step()
    
    cnn.eval()
    with torch.no_grad():
        preds = cnn(win_va_t).squeeze().numpy() * y_max
        pred_cnn[val_idx] = np.clip(preds, 0, None)

# Retrain full for visualization
train_mean = windows.mean(axis=0, keepdims=True)
train_std = windows.std(axis=0, keepdims=True) + 1e-8
win_full = (windows - train_mean) / train_std
y_max = true_error_spatial.max() + 1e-8
y_full_scaled = true_error_spatial / y_max

win_full_t = torch.tensor(win_full, dtype=torch.float32)
y_full_t = torch.tensor(y_full_scaled, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(win_full_t, y_full_t), batch_size=32, shuffle=True)

cnn_full = Conv1DNet(in_ch=r)
optimizer_cnn = optim.Adam(cnn_full.parameters(), lr=0.005)
cnn_full.train()
for _ in range(100):
    for bx, by in loader:
        optimizer_cnn.zero_grad()
        loss_c = nn.MSELoss()(cnn_full(bx), by)
        loss_c.backward()
        optimizer_cnn.step()

# ─────────────────────────────────────────────────────────────
# 5. EVALUATION: DMD-CNN vs PDE Residual
# ─────────────────────────────────────────────────────────────
corr_cnn = np.corrcoef(true_error_spatial, pred_cnn)[0, 1]
corr_res = np.corrcoef(true_error_spatial, residual_spatial)[0, 1]
mae_cnn = mean_absolute_error(true_error_spatial, pred_cnn)

def adaptive_refinement(pred, local_error, pct=75):
    mask = pred > np.percentile(pred, pct)
    e_before = local_error.copy()
    e_after = local_error.copy()
    e_after[mask] *= 0.38
    improvement = (e_before.mean() - e_after.mean()) / e_before.mean() * 100
    return mask, improvement

_, imp_cnn = adaptive_refinement(pred_cnn, true_error_spatial)
_, imp_res = adaptive_refinement(residual_spatial, true_error_spatial)

print("\n" + "="*70)
print("PHASE 4 METRICS (Allen-Cahn True PINN)")
print("="*70)
print(f"PDE Residual Correlation : {corr_res:.4f} | Refinement: {imp_res:.1f}%")
print(f"DMD+CNN Correlation      : {corr_cnn:.4f} | Refinement: {imp_cnn:.1f}%")
print("="*70)
if corr_cnn > corr_res:
    print("SUCCESS: Spectral DMD features predict local error BETTER than the raw physics residual!")
else:
    print("BASELINE WINS: The physics residual is more informative for this PDE.")

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0a0a14')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
TXT = 'white'

def style_ax(ax, title):
    ax.set_facecolor('#12121e')
    ax.set_title(title, color=TXT, fontsize=9, fontweight='bold', pad=10)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3a')

# Panel 0: True Solution
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(U_exact, aspect='auto', cmap='magma', origin='lower', vmin=-1, vmax=1)
ax.set_xlabel('Time', color=TXT)
ax.set_ylabel('Space', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Exact Allen-Cahn Solution')

# Panel 1: PINN Prediction
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(final_pinn, aspect='auto', cmap='magma', origin='lower', vmin=-1, vmax=1)
ax.set_xlabel('Time', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'PINN Final Prediction')

# Panel 2: True Absolute Error Heatmap
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(np.abs(final_pinn - U_exact), aspect='auto', cmap='Reds', origin='lower')
ax.set_xlabel('Time', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Absolute Error |PINN - True|')

# Panel 3: DMD Mode Fields (Top 5)
ax = fig.add_subplot(gs[1, 0])
for i in range(5):
    ax.plot(x, Phi_spatial[:, i], label=f'Mode {i+1}')
ax.set_xlabel('Space (x)', color=TXT)
ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=TXT)
style_ax(ax, 'Top 5 DMD Spatial Modes')

# Panel 4: Error Indicators (Normalized)
ax = fig.add_subplot(gs[1, 1:3])
norm_true = true_error_spatial / true_error_spatial.max()
norm_cnn = pred_cnn / pred_cnn.max()
norm_res = residual_spatial / residual_spatial.max()

ax.plot(x, norm_true, color='#ffd700', lw=2, label='True Error Profile', alpha=0.8)
ax.plot(x, norm_cnn, color='#00d4ff', lw=2, linestyle='--', label=f'DMD+CNN (r={corr_cnn:.3f})')
ax.plot(x, norm_res, color='#ff6b6b', lw=2, linestyle=':', label=f'PDE Residual (r={corr_res:.3f})')
ax.set_xlabel('Space (x)', color=TXT)
ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor=TXT)
style_ax(ax, 'Comparison: DMD+CNN vs PDE Residual')

fig.suptitle(
    "Spectral Error Indicators — Phase 4: Allen-Cahn Real PINN Dynamics",
    color=TXT, fontsize=12, fontweight='bold', y=0.96
)

out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'phase4_results.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 4 results figure saved → outputs/phase4_results.png")
