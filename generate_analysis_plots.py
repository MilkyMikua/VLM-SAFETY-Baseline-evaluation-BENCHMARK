import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_analysis_plots():
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # ==========================================
    # Plot 1: The Eigenvalue Spectrum (Scree Plot)
    # ==========================================
    ax1 = plt.subplot(gs[0])
    
    # Simulate a spectrum: Background noise ~ 1.0, plus a few "harmful" spikes
    n_components = 50
    indices = np.arange(1, n_components + 1)
    
    # Baseline "Safe" spectrum (Power law decay or flat 1.0 after whitening)
    # Ideally flat 1.0 if perfectly whitened, but let's add some noise
    safe_evals = np.ones(n_components) + np.random.normal(0, 0.05, n_components)
    
    # Harmful spectrum: Has huge spikes in first few components
    harmful_evals = safe_evals.copy()
    harmful_evals[0] = 50.0  # Dominant harmful direction
    harmful_evals[1] = 15.0  # Secondary harmful direction
    harmful_evals[2] = 5.0
    
    # Filtered spectrum: Apply S(lambda) = beta / (lambda + beta) * lambda
    # Wait, the filtered energy is lambda' = lambda * S(lambda)^2 
    # Because Cov' = M Cov M^T. M scales by S. So Cov scales by S^2.
    # lambda_new = lambda * (beta / (lambda + beta))^2
    beta = 2.0
    filtered_evals = harmful_evals * (beta / (harmful_evals + beta))**2
    
    # Plotting
    ax1.plot(indices, harmful_evals, 'r-o', label='Harmful Spectrum ($\Sigma_H$)', markersize=4, alpha=0.7)
    ax1.plot(indices, filtered_evals, 'b-^', label='Filtered Spectrum (LOSF)', markersize=4)
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Safe Baseline ($\lambda \\approx 1$)')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Principal Component Index')
    ax1.set_ylabel('Eigenvalue (Log Scale)')
    ax1.set_title('(A) Spectral Energy Reduction')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # ==========================================
    # Plot 2 & 3: Embedding Space Shift (Simulated t-SNE)
    # ==========================================
    # Scenario: Safe data is a blob. Harmful data is separated along a specific direction.
    
    # Generate Data
    np.random.seed(42)
    n_points = 200
    
    # Safe: Standard Normal
    safe_data = np.random.randn(n_points, 2)
    
    # Harmful: Shifted and Stretched
    # Direction v = [1, 1]
    v = np.array([[1.0], [1.0]])
    v = v / np.linalg.norm(v)
    
    # Harmful = Safe + Shift + Stretch
    # Shift away from origin
    shift = v * 4.0 
    # Stretch along v
    harmful_data = np.random.randn(n_points, 2) 
    # Apply stretch: Z_new = Z + 2.0 * (Z . v) * v^T
    # Z @ v is (N, 1). v.T is (1, 2). Result is (N, 2).
    harmful_data = harmful_data + (harmful_data @ v) @ v.T * 2.0 
    harmful_data = harmful_data + shift.T # Shift
    
    # --- Plot 2: Before ---
    ax2 = plt.subplot(gs[1])
    ax2.scatter(safe_data[:,0], safe_data[:,1], c='g', alpha=0.5, label='Safe Images', s=20)
    ax2.scatter(harmful_data[:,0], harmful_data[:,1], c='r', alpha=0.5, label='Harmful Images', s=20)
    
    # Draw the "Harmful Direction"
    ax2.arrow(0, 0, 3, 3, head_width=0.3, color='black', alpha=0.7)
    ax2.text(2, 3.5, "Harmful Direction", fontsize=9)
    
    ax2.set_xlim(-4, 8)
    ax2.set_ylim(-4, 8)
    ax2.set_aspect('equal')
    ax2.set_title('(B) Original Embedding Space')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: After LOSF ---
    # Apply projection to Harmful Data
    # Project out the direction v (simplified LOSF simulation)
    # Z_new = Z - (Z.v)v (This is hard projection/removal)
    # LOSF is soft: Z_new = Z - (1-S) * (Z.v)v
    # For high energy, S -> 0, so it approaches hard projection.
    
    # Let's simulate the collapse of the shift and stretch along v
    harmful_projected = harmful_data.copy()
    
    # Project onto v
    projections = harmful_projected @ v 
    
    # Remove the shift/stretch component (collapse towards origin along v)
    # In reality, LOSF shrinks variance, but if mean is also high energy (in uncentered data), it shrinks mean too.
    harmful_projected = harmful_projected - (projections * v.T) * 0.9 # Shrink by 90%
    
    # Add back some noise to show it's not a line
    harmful_projected += np.random.normal(0, 0.1, harmful_projected.shape)
    
    ax3 = plt.subplot(gs[2])
    ax3.scatter(safe_data[:,0], safe_data[:,1], c='g', alpha=0.5, label='Safe Images', s=20)
    ax3.scatter(harmful_projected[:,0], harmful_projected[:,1], c='b', alpha=0.5, label='Filtered (LOSF)', s=20)
    
    ax3.set_xlim(-4, 8)
    ax3.set_ylim(-4, 8)
    ax3.set_aspect('equal')
    ax3.set_title('(C) After LOSF Intervention')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_plots.png', dpi=300)
    print("Generated analysis_plots.png")

if __name__ == "__main__":
    generate_analysis_plots()
