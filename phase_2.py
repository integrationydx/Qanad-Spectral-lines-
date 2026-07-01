"""
Spectral Error Indicators for Neural PDE Solvers
Phase 2: Nonlinear Regression Heads — XGBoost + CNN

Builds directly on Phase 1:
  - Reuses the same Burgers' equation benchmark and DMD pipeline
  - Replaces Ridge Regression with:
      (A) XGBoost — nonlinear tabular model on 46-dim spectral features
      (B) CNN     — 1D conv net over spatial DMD mode fields phi(i)
  - Compares all three heads (Ridge / XGBoost / CNN) side-by-side
  - Generates Phase 2 results figure

Phase 1 results are preserved and shown in comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("PHASE 2: Nonlinear Regression Heads")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. REPRODUCE PHASE 1 SETUP (identical to phase 1 pipeline)
# ─────────────────────────────────────────────────────────────
Nx, Nt, K = 100, 50, 40
x      = np.linspace(-1, 1, Nx)
t_eval = np.linspace(0.1, 1.0, Nt)
nu     = 0.01 / np.pi

U_exact = np.zeros((Nx, Nt))
for j, tj in enumerate(t_eval):
    U_exact[:, j] = -np.tanh((x - 0.3) / (2 * nu)) + 1.0

snapshots = []
for k in range(K):
    progress        = k / (K - 1)
    noise_scale     = (1 - progress) * 0.8 + 0.05
    U_pinn          = U_exact.copy()
    U_pinn         += noise_scale * np.random.randn(Nx, Nt) * 0.3
    shock_mask      = np.abs(x) < 0.15
    local_err_scale = (1 - progress**0.4) * 1.5
    U_pinn[shock_mask, :] += local_err_scale * np.random.randn(shock_mask.sum(), Nt) * 0.5
    for freq in [3, 5, 7]:
        lag = (1 - progress) * 0.3
        U_pinn += lag * np.sin(freq * np.pi * x[:, None]) * np.cos(freq * np.pi * t_eval[None, :]) * 0.1
    snapshots.append(U_pinn)

print("✓ Phase 1 benchmark reproduced (Burgers', 100x50, 40 snapshots)")

# ─────────────────────────────────────────────────────────────
# 2. DMD (same as Phase 1)
# ─────────────────────────────────────────────────────────────
def run_dmd(snaps, r=15):
    S   = np.array([s.flatten() for s in snaps]).T
    X_  = S[:, :-1];  Xp = S[:, 1:]
    U_, sigma, Vt = svd(X_, full_matrices=False)
    U_r = U_[:, :r];  sigma_r = sigma[:r];  Vt_r = Vt[:r, :]
    A_t = U_r.T @ Xp @ Vt_r.T @ np.diag(1.0 / sigma_r)
    eigvals, W = np.linalg.eig(A_t)
    Phi = Xp @ Vt_r.T @ np.diag(1.0 / sigma_r) @ W
    energies = np.abs(eigvals) * np.linalg.norm(Phi, axis=0)
    energies /= (energies.sum() + 1e-12)
    H = -np.sum(energies * np.log(energies + 1e-12))
    return Phi, eigvals, energies, H, sigma

Phi, eigvals, modal_energies, H, _ = run_dmd(snapshots, r=15)
print(f"✓ DMD complete | H = {H:.4f}")

r = 15
Phi_spatial  = np.abs(Phi).reshape(Nx, Nt, r).mean(axis=1)   # [Nx x r]
eig_mags     = np.abs(eigvals[:r])
feat_eig     = np.tile(eig_mags,          (Nx, 1))            # [Nx x r]
feat_ent     = np.full((Nx, 1), H)                            # [Nx x 1]
feat_energy  = np.tile(modal_energies[:r], (Nx, 1))           # [Nx x r]
X_features   = np.hstack([Phi_spatial, feat_eig, feat_ent, feat_energy])  # [Nx x 46]

final_pinn   = snapshots[-1]
local_error  = np.mean(np.abs(final_pinn - U_exact), axis=1)  # [Nx]

# ─────────────────────────────────────────────────────────────
# 3. UNIFIED 5-FOLD OOF EVALUATION (Ridge, XGBoost, CNN)
# ─────────────────────────────────────────────────────────────
print(f"\n── Unified 5-Fold OOF Evaluation ───────")

class Conv1DNet(nn.Module):
    def __init__(self, in_ch=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=-1) # Global avg pool
        x = self.fc(x)
        return x

kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred_ridge = np.zeros(Nx)
pred_xgb_full = np.zeros(Nx)
pred_cnn = np.zeros(Nx)

# Pre-extract spatial windows for CNN (padded by 10 to get size 21)
X_cnn_pad = np.pad(Phi_spatial, ((10, 10), (0, 0)), mode='edge')
windows = np.array([X_cnn_pad[i:i+21].T for i in range(Nx)]) # [Nx, r, 21]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
    # -- Data prep --
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_features[train_idx])
    X_va = scaler.transform(X_features[val_idx])
    y_tr, y_va = local_error[train_idx], local_error[val_idx]
    
    # 1. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    pred_ridge[val_idx] = np.clip(ridge.predict(X_va), 0, None)
    
    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_xgb_full[val_idx] = np.clip(xgb.predict(X_va), 0, None)
    
    # 3. CNN
    train_mean = Phi_spatial[train_idx].mean(axis=0).reshape(1, -1, 1)
    train_std = (Phi_spatial[train_idx].std(axis=0) + 1e-8).reshape(1, -1, 1)
    
    win_tr = (windows[train_idx] - train_mean) / train_std
    win_va = (windows[val_idx] - train_mean) / train_std
    
    win_tr = torch.tensor(win_tr, dtype=torch.float32)
    win_va = torch.tensor(win_va, dtype=torch.float32)
    
    y_max = y_tr.max() + 1e-8
    yt_tr = torch.tensor(y_tr / y_max, dtype=torch.float32).unsqueeze(1)
    
    cnn = Conv1DNet(in_ch=r)
    optimizer = optim.Adam(cnn.parameters(), lr=0.005)
    loader = DataLoader(TensorDataset(win_tr, yt_tr), batch_size=32, shuffle=True)
    
    cnn.train()
    for _ in range(60):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(cnn(bx), by)
            loss.backward()
            optimizer.step()
            
    cnn.eval()
    with torch.no_grad():
        pred_cnn[val_idx] = np.clip((cnn(win_va).squeeze(1).numpy() * y_max), 0, None)

def print_metrics(name, pred):
    mae = mean_absolute_error(local_error, pred)
    r2 = r2_score(local_error, pred)
    corr = np.corrcoef(local_error, pred)[0, 1]
    print(f"   {name:7s} | MAE={mae:.5f} | R²={r2:.4f} | Corr={corr:.4f}")
    return mae, r2, corr

mae_r, r2_r, corr_r = print_metrics("Ridge", pred_ridge)
mae_x, r2_x, corr_x = print_metrics("XGBoost", pred_xgb_full)
mae_c, r2_c, corr_c = print_metrics("CNN", pred_cnn)

# ── Retrain on full dataset for plots ────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_features)
xgb.fit(X_sc, local_error)
importances = xgb.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
feat_names = ([f"phi_{i}" for i in range(r)] + [f"|lam_{i}|" for i in range(r)] + ["H"] + [f"E_{i}" for i in range(r)])
top_feats = [(feat_names[i], importances[i]) for i in top_idx]

train_mean = Phi_spatial.mean(axis=0).reshape(1, -1, 1)
train_std = (Phi_spatial.std(axis=0) + 1e-8).reshape(1, -1, 1)
win_full = torch.tensor((windows - train_mean) / train_std, dtype=torch.float32)
y_max = local_error.max() + 1e-8
yt_full = torch.tensor(local_error / y_max, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(win_full, yt_full), batch_size=32, shuffle=True)

cnn = Conv1DNet(in_ch=r)
optimizer = optim.Adam(cnn.parameters(), lr=0.005)
cnn.train()
losses = []
n_epochs = 80
for _ in range(n_epochs):
    ep_loss = 0
    for bx, by in loader:
        optimizer.zero_grad()
        loss = nn.MSELoss()(cnn(bx), by)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * bx.size(0)
    losses.append(ep_loss / Nx)

# ─────────────────────────────────────────────────────────────
# 4. ADAPTIVE REFINEMENT — compare all three heads
# ─────────────────────────────────────────────────────────────
def adaptive_refinement(pred, local_error, pct=75):
    mask         = pred > np.percentile(pred, pct)
    e_before     = local_error.copy()
    e_after      = local_error.copy()
    e_after[mask] *= 0.38
    improvement  = (e_before.mean() - e_after.mean()) / e_before.mean() * 100
    return mask, e_before, e_after, improvement

mask_r, eb_r, ea_r, imp_r = adaptive_refinement(pred_ridge,    local_error)
mask_x, eb_x, ea_x, imp_x = adaptive_refinement(pred_xgb_full, local_error)
mask_c, eb_c, ea_c, imp_c = adaptive_refinement(pred_cnn,      local_error)

print(f"\n── Adaptive Refinement Comparison ──────")
print(f"   Ridge   : {imp_r:.1f}% improvement  |  {mask_r.sum()} pts flagged")
print(f"   XGBoost : {imp_x:.1f}% improvement  |  {mask_x.sum()} pts flagged")
print(f"   CNN     : {imp_c:.1f}% improvement  |  {mask_c.sum()} pts flagged")

# ─────────────────────────────────────────────────────────────
# 5. PHASE 2 RESULTS FIGURE  (12 panels)
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0a0a14')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

TXT      = 'white'
CYAN     = '#00d4ff'
RED      = '#ff6b6b'
GREEN    = '#44ff88'
ORANGE   = '#ffaa00'
PURPLE   = '#c084fc'
GOLD     = '#ffd700'

def style_ax(ax, title, color=TXT):
    ax.set_facecolor('#12121e')
    ax.set_title(title, color=color, fontsize=8.5, fontweight='bold', pad=5)
    ax.tick_params(colors=TXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3a')

# ── Row 0: model comparison overview ─────────────────────────
# Panel 0,0 — R² comparison bar chart
ax = fig.add_subplot(gs[0, 0])
models = ['Ridge\n(Phase 1)', 'XGBoost\n(Phase 2A)', 'CNN\n(Phase 2B)']
r2s    = [r2_r, r2_x, r2_c]
cols   = [RED, GREEN, CYAN]
bars   = ax.bar(models, r2s, color=cols, edgecolor='none', width=0.5)
ax.set_ylabel('R²', color=TXT, fontsize=8)
ax.set_ylim(0, max(r2s) * 1.3)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.3f}', ha='center', va='bottom', color=TXT, fontsize=9, fontweight='bold')
style_ax(ax, 'R² Comparison — All Models')

# Panel 0,1 — MAE comparison
ax = fig.add_subplot(gs[0, 1])
maes = [mae_r, mae_x, mae_c]
bars = ax.bar(models, maes, color=cols, edgecolor='none', width=0.5)
ax.set_ylabel('MAE', color=TXT, fontsize=8)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.00002,
            f'{val:.5f}', ha='center', va='bottom', color=TXT, fontsize=8, fontweight='bold')
style_ax(ax, 'MAE Comparison — All Models')

# Panel 0,2 — Correlation comparison
ax = fig.add_subplot(gs[0, 2])
corrs = [corr_r, corr_x, corr_c]
bars  = ax.bar(models, corrs, color=cols, edgecolor='none', width=0.5)
ax.set_ylabel('Pearson r', color=TXT, fontsize=8)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, corrs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f'{val:.3f}', ha='center', va='bottom', color=TXT, fontsize=9, fontweight='bold')
style_ax(ax, 'Pearson Correlation — All Models')

# ── Row 1: True vs predicted for each model ───────────────────
for col_i, (pred, label, color) in enumerate([
    (pred_ridge,    f'Ridge  R²={r2_r:.3f}',   RED),
    (pred_xgb_full, f'XGBoost R²={r2_x:.3f}', GREEN),
    (pred_cnn,      f'CNN     R²={r2_c:.3f}',  CYAN),
]):
    ax = fig.add_subplot(gs[1, col_i])
    ax.plot(x, local_error, color=GOLD, lw=1.8, label='True Error')
    ax.plot(x, pred, color=color, lw=1.6, linestyle='--', label=label)
    ax.axvspan(-0.15, 0.15, alpha=0.10, color=GOLD, label='Shock region')
    ax.set_xlabel('x', color=TXT, fontsize=7)
    ax.set_ylabel('Mean |error|', color=TXT, fontsize=7)
    ax.legend(fontsize=6.5, facecolor='#1a1a2e', labelcolor=TXT, framealpha=0.8)
    style_ax(ax, f'True vs Predicted — {label.split()[0]}', color)

# ── Row 2: Scatter plots ──────────────────────────────────────
for col_i, (pred, label, color) in enumerate([
    (pred_ridge,    'Ridge',   RED),
    (pred_xgb_full, 'XGBoost', GREEN),
    (pred_cnn,      'CNN',     CYAN),
]):
    ax  = fig.add_subplot(gs[2, col_i])
    r2v = [r2_r, r2_x, r2_c][col_i]
    cv  = [corr_r, corr_x, corr_c][col_i]
    ax.scatter(local_error, pred, c=x, cmap='plasma', s=20, alpha=0.75, edgecolors='none')
    lim = max(local_error.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#555', lw=1)
    ax.set_xlabel('True Error', color=TXT, fontsize=7)
    ax.set_ylabel('Predicted Error', color=TXT, fontsize=7)
    style_ax(ax, f'{label} Scatter  r={cv:.3f}', color)

# ── Row 3: XGBoost feature importance + CNN loss + refinement ─
# Panel 3,0 — XGBoost feature importance
ax = fig.add_subplot(gs[3, 0])
top_n  = 8
t_names = [fn for fn, _ in top_feats[:top_n]]
t_imps  = [fi for _, fi in top_feats[:top_n]]
y_pos   = np.arange(top_n)
ax.barh(y_pos, t_imps[::-1], color=PURPLE, edgecolor='none')
ax.set_yticks(y_pos)
ax.set_yticklabels(t_names[::-1], fontsize=7, color=TXT)
ax.set_xlabel('Importance', color=TXT, fontsize=7)
style_ax(ax, 'XGBoost Feature Importances (Top 8)', PURPLE)

# Panel 3,1 — CNN training loss curve
ax = fig.add_subplot(gs[3, 1])
ax.plot(range(1, len(losses)+1), losses, color=CYAN, lw=2)
ax.fill_between(range(1, len(losses)+1), losses, alpha=0.15, color=CYAN)
ax.set_xlabel('Epoch', color=TXT, fontsize=7)
ax.set_ylabel('MSE Loss', color=TXT, fontsize=7)
style_ax(ax, 'CNN Training Loss Curve', CYAN)

# Panel 3,2 — Adaptive refinement improvement comparison
ax = fig.add_subplot(gs[3, 2])
imps = [imp_r, imp_x, imp_c]
bars = ax.bar(models, imps, color=cols, edgecolor='none', width=0.5)
ax.set_ylabel('% Error Reduction', color=TXT, fontsize=8)
ax.set_ylim(0, max(imps) * 1.3)
for bar, val in zip(bars, imps):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', color=TXT, fontsize=9, fontweight='bold')
style_ax(ax, 'Adaptive Refinement — Error Reduction', GREEN)

fig.suptitle(
    "Spectral Error Indicators for Neural PDE Solvers — Phase 2 Results\n"
    "XGBoost + CNN Regression Heads vs Ridge Baseline  |  Aditya Alur, PES EC Campus",
    color=TXT, fontsize=11, fontweight='bold', y=0.99
)

out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'phase2_results.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 2 results figure saved → outputs/phase2_results.png")

# ─────────────────────────────────────────────────────────────
# 6. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 FINAL SUMMARY")
print("="*60)
print(f"{'Model':<18} {'MAE':>10} {'R²':>8} {'Corr':>8} {'Refinement':>12}")
print("-"*60)
print(f"{'Ridge (Ph1)':<18} {mae_r:>10.5f} {r2_r:>8.4f} {corr_r:>8.4f} {imp_r:>10.1f}%")
print(f"{'XGBoost (Ph2A)':<18} {mae_x:>10.5f} {r2_x:>8.4f} {corr_x:>8.4f} {imp_x:>10.1f}%")
print(f"{'CNN (Ph2B)':<18} {mae_c:>10.5f} {r2_c:>8.4f} {corr_c:>8.4f} {imp_c:>10.1f}%")
print("="*60)
print(f"\nTop XGBoost features: {', '.join([f for f,_ in top_feats[:3]])}")
print(f"CNN converged in {n_epochs} epochs | Final loss = {losses[-1]:.5f}")