import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Data extracted from JSONs
# Format: Method: {Metric: Value}
data = {
    "LEACE": {
        "Utility Recall (Pre)": 0.873, "Utility Recall (Post)": 0.873,
        "Harmful Recall (Pre)": 0.935, "Harmful Recall (Post)": 0.935,
        "CLIPScore (Safe)": 0.260
    },
    "Safe-CLIP": {
        "Utility Recall (Pre)": 0.778, "Utility Recall (Post)": 0.468,
        "Harmful Recall (Pre)": 0.764, "Harmful Recall (Post)": 0.394,
        "CLIPScore (Safe)": 0.187
    },
    "ETA": {
        "Utility Recall (Pre)": 0.855, "Utility Recall (Post)": 0.772,
        "Harmful Recall (Pre)": 0.889, "Harmful Recall (Post)": 0.847,
        "CLIPScore (Safe)": 0.282
    },
    "LOSF (Ours)": {
        "Utility Recall (Pre)": 0.778, "Utility Recall (Post)": 0.770,
        "Harmful Recall (Pre)": 0.764, "Harmful Recall (Post)": 0.758,
        "CLIPScore (Safe)": 0.329
    }
}

def plot_bar_comparison():
    methods = list(data.keys())
    
    # Metrics to plot
    u_pre = [data[m]["Utility Recall (Pre)"] for m in methods]
    u_post = [data[m]["Utility Recall (Post)"] for m in methods]
    h_pre = [data[m]["Harmful Recall (Pre)"] for m in methods]
    h_post = [data[m]["Harmful Recall (Post)"] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting Utility Retention
    # We want to show Trade-off. 
    # Let's plot Post-Utility vs Post-Safety (where Safety = 1 - Harmful Recall)?
    # Or just raw bars.
    
    rects1 = ax.bar(x - width/2, u_post, width, label='Utility Recall (Safe)', color='#4caf50')
    rects2 = ax.bar(x + width/2, h_post, width, label='Harmful Recall (Unsafe)', color='#e53935')
    
    # Add Pre markers as lines? Or just focus on Post.
    # The user wants to see the method's effect. 
    # Let's add dashed lines for "Pre" levels.
    # But Pre varies per method (different base models/sets).
    
    for i, (up, hp) in enumerate(zip(u_pre, h_pre)):
        ax.hlines(up, x[i]-width/2, x[i], colors='green', linestyles='dashed', alpha=0.5)
        ax.hlines(hp, x[i], x[i]+width/2, colors='red', linestyles='dashed', alpha=0.5)

    ax.set_ylabel('Recall@K')
    ax.set_title('Utility vs Harmful Recall (Post-Intervention)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # Add text for retention/drop
    def autolabel(rects, pres):
        for rect, pre in zip(rects, pres):
            height = rect.get_height()
            change = ((height - pre) / pre) * 100
            ax.annotate(f'{height:.2f}\n({change:+.1f}%)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1, u_pre)
    autolabel(rects2, h_pre)
    
    fig.tight_layout()
    plt.savefig('comparison_chart.png', dpi=300)
    print("Generated comparison_chart.png")

def plot_concept_diagram():
    # A schematic showing 2D Gaussian distributions
    # Safe (Green), Harmful (Red)
    # LOSF transforms them.
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Background
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw Safe Distribution (Standard Normal after Whitening)
    safe_circle = patches.Circle((0, 0), 1.0, edgecolor='green', facecolor='none', linestyle='--', linewidth=2, label='Safe (Whitened)')
    ax.add_patch(safe_circle)
    
    # Draw Harmful Distribution (Stretched along one axis)
    # Ellipse rotated 45 degrees
    angle = 45
    width = 2.5 # High energy
    height = 1.0 # Normal energy
    harm_ellipse = patches.Ellipse((0, 0), width, height, angle=angle, edgecolor='red', facecolor='none', linewidth=2, label='Harmful')
    ax.add_patch(harm_ellipse)
    
    # Draw LOSF Shrinkage Direction
    # Arrow along the major axis
    dx = np.cos(np.radians(angle)) * (width/2)
    dy = np.sin(np.radians(angle)) * (width/2)
    ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
    ax.text(dx+0.1, dy+0.1, 'Harmful Eigen-direction', color='red', fontsize=10)
    
    # Draw LOSF Filtered Harmful (Shrunk)
    shrunk_width = 1.0 # Shrunk back to safe level
    harm_shrunk = patches.Ellipse((0, 0), shrunk_width, height, angle=angle, edgecolor='blue', facecolor='none', linestyle='-', linewidth=2, label='Filtered (LOSF)')
    ax.add_patch(harm_shrunk)
    
    # Legend
    ax.legend(loc='upper left')
    ax.set_title("Concept: LOSF Spectral Shrinkage\n(In Whitened Safe Space)", fontsize=14)
    
    plt.savefig('concept_diagram.png', dpi=300)
    print("Generated concept_diagram.png")

if __name__ == "__main__":
    plot_bar_comparison()
    plot_concept_diagram()
