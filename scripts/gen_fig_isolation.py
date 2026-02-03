#!/usr/bin/env python3
"""Generate Figure 9: noise source isolation α summary."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'serif',
})

# QPU data from idle_qubit_experiment (Rigetti Ankaa-3, 10k shots each)
circuits = ['Idle\n(I gates)', 'Hadamard\n(H gates)', 'Bell\npairs', 'Full\nchain']
alphas = [-0.964, -0.927, -1.032, -1.309]
ses    = [ 0.033,  0.018,  0.027,  0.032]
p_vals = [ 0.298,  0.00115, 0.252, 8.47e-8]
colors = ['#999999', '#2ca02c', '#9467bd', '#1f77b4']

fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0))  # single-column PRA width

x = np.arange(len(circuits))
bars = ax.bar(x, alphas, yerr=ses, capsize=4, color=colors, alpha=0.75,
              edgecolor='black', linewidth=0.6, width=0.7)
ax.axhline(-1.0, color='red', ls='--', lw=1.5, alpha=0.6,
           label=r'i.i.d.: $\alpha = -1.0$')
ax.axhline(0.0, color='black', ls='-', lw=0.3, alpha=0.3)

# Add significance annotations
for i, (a, p) in enumerate(zip(alphas, p_vals)):
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'
    y_off = -0.07 if a < -1.0 else 0.03
    ax.text(i, a + y_off, sig, ha='center', va='bottom' if a > -1.0 else 'top',
            fontsize=8, fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(circuits, ha='center', fontsize=8)
ax.set_ylabel(r'$\alpha$ (Var($H$))')
ax.set_ylim(-1.5, 0.05)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
fig.savefig('data/vacuum_telescope_v1/figures/figure_9_isolation.png',
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: data/vacuum_telescope_v1/figures/figure_9_isolation.png")

