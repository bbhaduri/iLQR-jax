import jax
import jax.numpy as jnp

from matplotlib import pyplot as plt

def plot_task_state(pos, targets, axs):
    """
    """
    targets_cm = targets * 100 #cm

    for ax in axs:
        ax.clear()

    # Convert positions to cm
    pos_cm = pos*100

    def plot_arm(ax, i):
        # Target locations
        ax.scatter(targets_cm[:, 0], targets_cm[:, 1], marker='s', color='green', s=200) #TODO, figure out how big these need to be

        # Start Locations
        ax.scatter(0., 0., color='red') # Shoulder
        ax.scatter(pos_cm[i, 0, 0], pos_cm[i, 0, 1], color='blue') # Hand
        ax.scatter(pos_cm[i, 1, 0], pos_cm[i, 1, 1], color='green') # Elbow

        # Make arms by connecting joints
        ax.plot([0., pos_cm[i, 1, 0]], [0., pos_cm[i, 1, 1]], color='blue') # Lower Arm
        ax.plot([pos_cm[i, 0, 0], pos_cm[i, 1, 0]], [pos_cm[i, 0, 1], pos_cm[i, 1, 1]], color='blue') # Upper Arm

        # Set axis boundaries
        ax.set_xlim([-32.5, 32.5])
        ax.set_ylim([-5, 60])

        return ax
    
    return [plot_arm(axs[i], i) for i in range(len(axs))]
    
# Plot eigenvalues of Matrix
def plot_eigenvalues(Uh, title=None, return_eigs=False):
    """
    """
    # Calculate eigenvalues
    eigenvalues = jnp.linalg.eigvals(Uh)
    
    # Plot the eigenvalues
    plt.figure(figsize=(8, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue')
    if title is not None:
        plt.title(f'{title} Eigenvalues')
    else:
        plt.title('Eigenvalues of Matrix')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.show()

    if return_eigs:
        return eigenvalues