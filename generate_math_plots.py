import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def plot_spectral_filter_curve():
    """
    Plots the shrinkage factor S(lambda) = beta / (lambda + beta)
    across a range of eigenvalues lambda.
    """
    lambdas = np.logspace(-1, 3, 500) # 0.1 to 1000
    betas = [0.1, 1.0, 5.0, 10.0]
    
    plt.figure(figsize=(10, 6))
    
    for beta in betas:
        S = beta / (lambdas + beta)
        plt.plot(lambdas, S, label=f'$\\beta={beta}$', linewidth=2.5)
        
    # Annotations
    plt.xscale('log')
    plt.xlabel(r'Eigenvalue $\lambda$ (Energy in Whitened Space)', fontsize=12)
    plt.ylabel(r'Shrinkage Factor $S(\lambda)$', fontsize=12)
    plt.title(r'Spectral Shrinkage Function: $S(\lambda) = \frac{\beta}{\lambda + \beta}$', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Highlight Regions
    plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
    plt.text(1.1, 0.6, 'Safe Subspace\n($\lambda \\approx 1$)', color='green', fontsize=10)
    
    plt.axvline(x=100.0, color='red', linestyle='--', alpha=0.5)
    plt.text(110, 0.6, 'Harmful Subspace\n($\lambda \\gg 1$)', color='red', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('math_plot_spectral_filter.png', dpi=300)
    print("Generated math_plot_spectral_filter.png")

def plot_energy_ellipsoids():
    """
    Visualizes the transformation of the covariance ellipsoid.
    We show 2D cross-section.
    Safe Distribution: Identity Circle.
    Harmful Distribution: Elongated Ellipse.
    Filtered: Harmful Ellipse shrunk along major axis.
    """
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Safe: Unit Circle (Whitened)
    x_safe = np.cos(theta)
    y_safe = np.sin(theta)
    
    # Harmful: Covariance Sigma_h. Eigenvalues [lambda1, lambda2]
    # Let lambda1 = 20 (Harmful), lambda2 = 1 (Safe overlap)
    l1, l2 = 20.0, 1.0
    # Rotation 45 degrees
    angle = np.pi/4
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    # Ellipse points in eigenbasis
    xe_h = np.sqrt(l1) * np.cos(theta)
    ye_h = np.sqrt(l2) * np.sin(theta)
    pts_h = np.stack([xe_h, ye_h])
    
    # Rotate to ambient space
    pts_h_rot = R @ pts_h
    
    # Filtered: Apply S(lambda)
    # beta = 5
    beta = 5.0
    s1 = beta / (l1 + beta) # 5/25 = 0.2
    s2 = beta / (l2 + beta) # 5/6 = 0.83
    
    xe_f = np.sqrt(l1) * s1 * np.cos(theta) # Note: Shrink amplitude by S? 
    # Wait, Covariance shrinks by S^2? Or representation z shrinks by S?
    # z' = S * z. So the cloud shrinks by S.
    # So boundary is s * sqrt(lambda).
    
    ye_f = np.sqrt(l2) * s2 * np.sin(theta)
    pts_f = np.stack([xe_f, ye_f])
    pts_f_rot = R @ pts_f
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Safe
    ax.plot(x_safe, y_safe, 'g--', label='Safe Distribution ($\Sigma=I$)', linewidth=2)
    ax.fill(x_safe, y_safe, 'g', alpha=0.05)
    
    # Plot Harmful
    ax.plot(pts_h_rot[0], pts_h_rot[1], 'r-', label=f'Harmful Distribution ($\lambda_1={l1}$)', linewidth=2)
    
    # Plot Filtered
    ax.plot(pts_f_rot[0], pts_f_rot[1], 'b-', label=f'Filtered (LOSF, $\\beta={beta}$)', linewidth=2)
    ax.fill(pts_f_rot[0], pts_f_rot[1], 'b', alpha=0.1)
    
    # Draw Eigenvectors
    v1 = R @ np.array([1, 0]) * np.sqrt(l1)
    ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, color='red', alpha=0.5)
    ax.text(v1[0], v1[1], r'$v_1$ (Harmful)', color='red')
    
    # Draw Filtered Vector
    v1_f = R @ np.array([1, 0]) * np.sqrt(l1) * s1
    ax.arrow(0, 0, v1_f[0], v1_f[1], head_width=0.2, color='blue')
    ax.text(v1_f[0]+0.2, v1_f[1], r"$v_1'$", color='blue')
    
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True)
    ax.set_title("Geometric View: Selective Spectral Shrinkage")
    ax.legend(loc='upper right')
    
    plt.savefig('math_plot_geometry.png', dpi=300)
    print("Generated math_plot_geometry.png")

if __name__ == "__main__":
    plot_spectral_filter_curve()
    plot_energy_ellipsoids()
