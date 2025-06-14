import jax
from jax import numpy as np
from jax.lax import scan
from jax import vmap, jit

# Parameters
N = 100 # cm->m
L1 = 30 / N  # length of upper arm
L2 = 33 / N# length of lower arm link
D2 = 16 /N# center of mass lower link, away from elbow
d = 20 / N # reach distance

M1 = 1.4 # mass of upper arm link
M2 = 1.0 # mass of lower arm link
I1 = 0.025  # moment of inerta upper link
I2 = 0.045 # moment of intertia lower link

# Constants for model
a1 = I1 + I2 + M2 * L1**2 
a2 = M2 * L1 * D2
a3 = I2

# System state
theta0 = np.array([10, 143.54]) * np.pi/180
x0 = np.concatenate((theta0, np.zeros((2, ))))

# step size
dt = .01

def get_M(theta):
    c = np.cos(theta[1])
    return N * np.array([[a1 + 2*a2*c, a3 + a2*c], 
                     [a3 + a2*c, a3]])

def get_Chi(theta, thetadot):
    Chi = np.array([-thetadot[1]*(2*thetadot[0]+thetadot[1]), thetadot[0]**2])
    Chi *= a2 * np.sin(theta[1]) 
    return Chi * N

B = np.array([[0.05, 0.025], [0.025, 0.05]]) * N

def continuous_dynamics(x, m):
    """
    dx/dt = f(x, m)
    x: [theta1, theta2, thetadot1, thetadot2]
    u: [u1, u2]
    """
    theta, thetadot = x[:2], x[2:]
    Minv = np.linalg.inv(get_M(theta))
    Chi = get_Chi(theta, thetadot)
    return np.concatenate((thetadot, Minv@(m - Chi -B@thetadot)))
    
def discrete_dynamics(x, m):
    """
    x[t+1] = f(x[t], m[t])
    Fw Euler approximation of continuous dynamics
    """
    xnew = x + dt*continuous_dynamics(x, m)
    return xnew, xnew

def get_angles(pos):
    """
    Calculate the angles of the arm segments given the hand position.
    Args:
        pos (jnp.ndarray): Hand position [x, y]
    Returns:
        jnp.ndarray: Joint angles [theta1, theta2]
    """
    x, y = pos
    theta1 = np.arctan2(y, x) - np.arccos((x**2 + y**2 + L1**2 - L2**2)
                                                     /(2*L1*(x**2 + y**2)**0.5))
    theta2 = np.arccos((x**2 + y**2 - L1**2 - L2**2)/(2*L1*L2))

    return theta1, theta2

# Define function for calculating arm positions
def get_arm_pos(thetas):
    """
    Calculate the positions of the both arm segments given the joint angles.
    Args:
        thetas (jnp.ndarray): Joint angles [theta1, theta2]
    Returns:
        jnp.ndarray: Positions of the hand and elbow
    """
    # Calculate positions and return
    elbow_pos = L1 * np.array([
        (np.cos(thetas[0])), (np.sin(thetas[0]))
    ])

    hand_pos = np.array(
        [(elbow_pos[0] + L2*np.cos(thetas.sum())),
         (elbow_pos[1] + L2*np.sin(thetas.sum()))]
    )

    return np.vstack([hand_pos, elbow_pos])

def get_position(x, full_arm=False):
    """
    Compute hand position from state
    """
    theta = x[:2]
    if full_arm:
        return get_arm_pos(theta)
    else:
        y1 = L1 * np.array([np.cos(theta[0]), np.sin(theta[0])])
        y2 = L2 * np.array([np.cos(theta.sum()), np.sin(theta.sum())])

        return y1 + y2

# Get position over entire trajectory (time steps, 4)
get_position_trj = vmap(get_position, in_axes=(0, None))
# Position over multiple trials (batch size, time steps, 4)
get_position_batch  = vmap(get_position_trj, in_axes=(0, None))

def rollout(x0, m_trj):
    """
    x0: (4, )
    m_trj: (time steps, 2)
    """
    time_steps = m_trj.shape[0]
    _, x_trj = scan(discrete_dynamics, x0, m_trj)
    y_trj = get_position_trj(x_trj)
    return x_trj, y_trj

rollout_batch = jit(vmap(rollout, in_axes=(None, 0)))