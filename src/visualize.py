import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Results from PEREGRINE rotation invariance test
ROTATIONS = [0, 15, 30, 45, 60, 90]
CAPSNET =   [94.0, 80.5, 55.0, 29.5, 20.5, 14.5]
RESNET =    [88.0, 80.0, 51.0, 26.0, 12.5, 10.5]
DELTA =     [c - r for c, r in zip(CAPSNET, RESNET)]

def plot_rotation_benchmark():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Plot 1 — Accuracy comparison
    x = np.arange(len(ROTATIONS))
    width = 0.35

    bars1 = ax1.bar(x - width/2, CAPSNET, width,
                    label='CapsNet (PEREGRINE)',
                    color='#1f6feb', alpha=0.9)
    bars2 = ax1.bar(x + width/2, RESNET, width,
                    label='ResNet Baseline',
                    color='#da3633', alpha=0.9)

    ax1.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('PEREGRINE — CapsNet vs ResNet\nRotation Invariance Benchmark',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r}°' for r in ROTATIONS])
    ax1.legend(facecolor='#21262d', labelcolor='white')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.2, color='white')

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 1,
                 f'{bar.get_height():.1f}%',
                 ha='center', va='bottom',
                 color='white', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 1,
                 f'{bar.get_height():.1f}%',
                 ha='center', va='bottom',
                 color='white', fontsize=8)

    # Plot 2 — Delta chart
    colors = ['#3fb950' if d > 0 else '#da3633' for d in DELTA]
    bars3 = ax2.bar(x, DELTA, color=colors, alpha=0.9)
    ax2.axhline(y=0, color='white', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Accuracy Delta (%)', fontsize=12)
    ax2.set_title('CapsNet Advantage Over ResNet\nby Rotation Angle',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{r}°' for r in ROTATIONS])
    ax2.grid(axis='y', alpha=0.2, color='white')

    for bar, delta in zip(bars3, DELTA):
        ax2.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.2,
                 f'+{delta:.1f}%',
                 ha='center', va='bottom',
                 color='white', fontsize=9,
                 fontweight='bold')

    caps_patch = mpatches.Patch(
        color='#3fb950',
        label='CapsNet wins — spatial relationships preserved'
    )
    ax2.legend(handles=[caps_patch],
               facecolor='#21262d',
               labelcolor='white')

    plt.tight_layout(pad=3.0)

    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/rotation_benchmark.png',
                dpi=150,
                bbox_inches='tight',
                facecolor='#0d1117')
    print("Chart saved to docs/rotation_benchmark.png")
    plt.show()


if __name__ == "__main__":
    print("Generating PEREGRINE rotation benchmark visualization...")
    plot_rotation_benchmark()
    print("Done.")
