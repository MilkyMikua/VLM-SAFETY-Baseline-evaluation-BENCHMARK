import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Helper for boxes
    def draw_box(x, y, w, h, text, color='#e3f2fd', ec='#2196f3', label_size=10):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=ec, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=label_size, wrap=True)
        return x+w, y+h/2 # Return attachment point (right)

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#546e7a'))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', fontsize=9, color='#37474f')

    # --- Phase 1: Offline Parameter Estimation (Top) ---
    ax.text(1, 7.5, "Phase 1: Offline Parameter Estimation", fontsize=12, fontweight='bold', color='#1565c0')
    
    # Safe Data Path
    draw_box(0.5, 6, 2, 1, "Safe Data\n(COCO/SBU)")
    draw_arrow(2.7, 6.5, 3.5, 6.5)
    draw_box(3.5, 6, 2, 1, "Compute\nWhitening (W)")
    
    # Harmful Data Path
    draw_box(6.5, 6, 2, 1, "Harmful Data\n(HatefulMemes)", color='#ffebee', ec='#ef5350')
    draw_arrow(8.7, 6.5, 9.5, 6.5)
    draw_box(9.5, 6, 3, 1, "Compute Harmful Stats\n(Covariance $\Sigma_H$)", color='#ffebee', ec='#ef5350')
    
    # Combine to Filter Params
    draw_arrow(5.7, 6.5, 6.5, 5.5, "") # W down to params
    draw_arrow(11.0, 6.0, 8.5, 5.5, "") # Stats down to params
    
    # Filter Params Box
    draw_box(6.0, 4.5, 3, 1, "Filter Parameters\n$U$ (Eigenvectors)\n$\lambda$ (Eigenvalues)", color='#fff3e0', ec='#ff9800')

    # --- Phase 2: Online Inference (Bottom) ---
    ax.text(1, 3.5, "Phase 2: Online Inference (LOSF Intervention)", fontsize=12, fontweight='bold', color='#2e7d32')
    
    # Main Pipeline
    # Image -> Vision Encoder -> ...
    draw_box(0.5, 2, 1.5, 1, "Input\nImage")
    draw_arrow(2.2, 2.5, 3.0, 2.5)
    
    draw_box(3.0, 2, 2.5, 1, "Vision Encoder\n(Layers $1 \\dots L$)")
    draw_arrow(5.7, 2.5, 6.5, 2.5, "Activations $Z$")
    
    # LOSF Operation Block (Detailed)
    # Drawing a large container for LOSF
    losf_rect = patches.Rectangle((6.5, 1.5), 5.0, 2.0, linewidth=2, edgecolor='#2e7d32', facecolor='#e8f5e9', linestyle='--')
    ax.add_patch(losf_rect)
    ax.text(6.6, 3.2, "LOSF Layer", fontsize=10, fontweight='bold', color='#2e7d32')
    
    # Feed params into LOSF
    draw_arrow(7.5, 4.5, 7.5, 3.5, "Inject Params")
    
    # Inside LOSF steps
    # 1. Whiten
    ax.text(7.2, 2.5, "$Z \\cdot W^T$", fontsize=9, bbox=dict(boxstyle="square", fc="white"))
    draw_arrow(7.8, 2.5, 8.2, 2.5)
    # 2. Shrink
    ax.text(8.3, 2.5, "Shrink\n$S(\\lambda)$", fontsize=9, bbox=dict(boxstyle="square", fc="#fff9c4"))
    draw_arrow(9.1, 2.5, 9.5, 2.5)
    # 3. Un-whiten
    ax.text(9.6, 2.5, "$\\cdot W^{-T}$", fontsize=9, bbox=dict(boxstyle="square", fc="white"))
    
    draw_arrow(11.5, 2.5, 12.0, 2.5, "$Z_{safe}$")
    
    draw_box(12.0, 2, 1.5, 1, "LLM\nDecoder")
    
    # --- Legend / Notes ---
    ax.text(7, 0.5, "Key Operation: $Z_{safe} = Z W^T [U S(\\lambda) U^T] W^{-T}$", 
            ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    plt.tight_layout()
    plt.savefig('pipeline_diagram.png', dpi=300)
    print("Generated pipeline_diagram.png")

if __name__ == "__main__":
    draw_pipeline()
