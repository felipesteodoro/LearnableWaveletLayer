"""Generate sample signal + DWT Db4 decomposition figures for Ford-A, SCP1, UWaveGesture."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
from pathlib import Path
from aeon.datasets import load_classification

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
OUT = Path('/Users/fteodoro/Dropbox/Doutorado/Tese/figuras')


def plot_dwt_panels(ax_list, signal, label, color, level=3, wavelet='db4'):
    """Plot signal + DWT levels in stacked axes. ax_list has len = level+1."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)  # [cA_L, cD_L, ..., cD_1]
    data = [signal] + list(coeffs)
    sub_labels = [label] + [f'$cA_{{{level}}}$'] + [f'$cD_{{{level-i}}}$' for i in range(level)]
    N = len(signal)
    for i, ax in enumerate(ax_list):
        d = data[i]
        x = np.linspace(0, N - 1, len(d))
        ax.plot(x, d, color=color, lw=0.85)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.5)
        ax.set_xlim(0, N - 1)
        ax.set_ylabel(sub_labels[i], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, ls=':')
        if i < len(ax_list) - 1:
            ax.set_xticklabels([])


# ============================================================================
# FORD-A — univariada, binária
# ============================================================================
print('Loading Ford-A...')
X_train, y_train = load_classification("FordA", split='train')
X_train = X_train.squeeze()
print(f'  Shape: {X_train.shape}, classes: {np.unique(y_train)}')

idx_neg = np.where(y_train == np.unique(y_train)[0])[0][0]  # Normal/Anomaly
idx_pos = np.where(y_train == np.unique(y_train)[1])[0][0]
sig_neg, sig_pos = X_train[idx_neg], X_train[idx_pos]

fig = plt.figure(figsize=(13, 7.5))
fig.suptitle('Ford-A — sinal e decomposição DWT (Db4, 3 níveis)',
             fontsize=12, fontweight='bold', y=0.995)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.18, wspace=0.20,
                       left=0.06, right=0.98, bottom=0.06, top=0.94)
axL = [fig.add_subplot(gs[i, 0]) for i in range(4)]
axR = [fig.add_subplot(gs[i, 1]) for i in range(4)]

plot_dwt_panels(axL, sig_neg, f'Sinal (classe {np.unique(y_train)[0]})', '#1976D2')
plot_dwt_panels(axR, sig_pos, f'Sinal (classe {np.unique(y_train)[1]})', '#C62828')

axL[-1].set_xlabel('Amostra temporal $t$', fontsize=9)
axR[-1].set_xlabel('Amostra temporal $t$', fontsize=9)

fig.savefig(OUT / 'fig_forda_samples.pdf', dpi=150, bbox_inches='tight')
print(f'  Saved {OUT / "fig_forda_samples.pdf"}')
plt.close()


# ============================================================================
# SCP1 — 6 canais EEG, binária
# ============================================================================
print('Loading SCP1...')
X_train, y_train = load_classification("SelfRegulationSCP1", split='train')
print(f'  Shape: {X_train.shape}, classes: {np.unique(y_train)}')
# X_train: (n_samples, n_channels, n_timesteps) → (227, 6, 896)

idx_a = np.where(y_train == np.unique(y_train)[0])[0][0]
idx_b = np.where(y_train == np.unique(y_train)[1])[0][0]
sig_a = X_train[idx_a]  # (6, 896)
sig_b = X_train[idx_b]

# Layout: 6 canais raw (à esquerda) + DWT do canal 0 (à direita)
fig = plt.figure(figsize=(13, 8))
fig.suptitle('SCP1 — registro EEG (6 canais) e decomposição DWT (Db4, 3 níveis, canal 1)',
             fontsize=12, fontweight='bold', y=0.995)
gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.18, wspace=0.20,
                       left=0.07, right=0.98, bottom=0.06, top=0.94,
                       width_ratios=[1, 1])

# Esquerda: 6 canais raw, amostra de cada classe sobreposta
for c in range(6):
    ax = fig.add_subplot(gs[c, 0])
    ax.plot(sig_a[c], color='#1976D2', lw=0.7, alpha=0.85,
            label=f'classe {np.unique(y_train)[0]}' if c == 0 else None)
    ax.plot(sig_b[c], color='#C62828', lw=0.7, alpha=0.85,
            label=f'classe {np.unique(y_train)[1]}' if c == 0 else None)
    ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.5)
    ax.set_ylabel(f'Canal {c+1}', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, ls=':')
    ax.set_xlim(0, sig_a.shape[1] - 1)
    if c == 0:
        ax.legend(fontsize=7, loc='upper right', ncol=2)
    if c < 5:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Amostra temporal $t$', fontsize=9)

# Direita: DWT do canal 0 (apenas classe A para clareza)
axR = [fig.add_subplot(gs[i, 1]) for i in range(4)]
plot_dwt_panels(axR, sig_a[0], f'Sinal canal 1 (classe {np.unique(y_train)[0]})', '#1976D2')
axR[-1].set_xlabel('Amostra temporal $t$', fontsize=9)
# Coloca anotação de qual classe a DWT está sendo aplicada
axR[0].set_title('Decomposição DWT (canal 1)', fontsize=9, loc='left', pad=4)

# Esconde linhas vazias do gs[4,1] e gs[5,1]
ax_empty1 = fig.add_subplot(gs[4, 1]); ax_empty1.axis('off')
ax_empty2 = fig.add_subplot(gs[5, 1]); ax_empty2.axis('off')

fig.savefig(OUT / 'fig_scp1_samples.pdf', dpi=150, bbox_inches='tight')
print(f'  Saved {OUT / "fig_scp1_samples.pdf"}')
plt.close()


# ============================================================================
# UWAVE-GESTURE — 3 canais accel (x/y/z), 8 classes
# ============================================================================
print('Loading UWaveGesture...')
X_train, y_train = load_classification("UWaveGestureLibrary", split='train')
print(f'  Shape: {X_train.shape}, classes: {np.unique(y_train)}')
# X_train: (n, 3, 315)

# Escolhe 3 gestos distintos: classes 1, 4, 8 (primeiros)
classes_to_show = np.unique(y_train)[[0, 3, 6]]  # 3 gestos representativos
print(f'  Showing classes: {classes_to_show}')

fig = plt.figure(figsize=(13, 9))
fig.suptitle('UWaveGesture — leituras de acelerômetro (3 eixos) e decomposição DWT (Db4, 3 níveis, eixo Z)',
             fontsize=12, fontweight='bold', y=0.995)
# Layout: 3 linhas (1 por classe), 7 colunas (3 raw + 4 DWT)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.18,
                       left=0.06, right=0.98, bottom=0.06, top=0.94,
                       width_ratios=[1, 1])

channel_names = ['Accel-X', 'Accel-Y', 'Accel-Z']
colors_xyz = ['#1976D2', '#2E7D32', '#C62828']

for row, cls in enumerate(classes_to_show):
    idx = np.where(y_train == cls)[0][0]
    sig = X_train[idx]  # (3, 315)

    # Painel esquerdo: 3 canais sobrepostos
    gs_left = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[row, 0], hspace=0.10)
    for c in range(3):
        ax = fig.add_subplot(gs_left[c])
        ax.plot(sig[c], color=colors_xyz[c], lw=1.0)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.5)
        ax.set_ylabel(channel_names[c], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, ls=':')
        ax.set_xlim(0, sig.shape[1] - 1)
        if c == 0:
            ax.set_title(f'Gesto {cls}', fontsize=9, loc='left', pad=4)
        if c < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Amostra temporal $t$', fontsize=8)

    # Painel direito: DWT do eixo Z
    gs_right = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[row, 1], hspace=0.10)
    axR = [fig.add_subplot(gs_right[i]) for i in range(4)]
    plot_dwt_panels(axR, sig[2], f'Accel-Z (gesto {cls})', '#C62828')
    axR[-1].set_xlabel('Amostra temporal $t$', fontsize=8)

fig.savefig(OUT / 'fig_uwave_samples.pdf', dpi=150, bbox_inches='tight')
print(f'  Saved {OUT / "fig_uwave_samples.pdf"}')
plt.close()

print('\nTodas as figuras geradas com sucesso.')
