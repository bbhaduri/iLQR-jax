import jax
import jax.random as jr
import numpy as onp

from jax import numpy as np
from jax.lax import scan
from jax import vmap, jit

import pickle

import arm_model

# OLD Network dynamics
# with open("../data/network_s972356.pickle", 'rb') as handle:
#     data = pickle.load(handle)
# params = data['params']
# C = np.asarray(params['C'])
# W = np.asarray(data['W'])
# hbar = np.asarray(data['hbar'])

# Parameters from Schimel et al.
W = np.asarray(onp.fromfile("../data/w", sep=' ')).reshape(200, 200)
C = np.asarray(onp.fromfile("../data/c", sep=' ')).reshape(2, 200).T
hbar = 20. + 5. * jr.normal(jr.PRNGKey(0), (200,))

# Activation function
phi = lambda x: jax.nn.relu(x)
# step size
dt = .001
# Network time constant
tau = .150

def continuous_network_dynamics(x, inputs):
    return (-x + W.dot(phi(x)) + inputs + hbar) / tau

def discrete_network_dynamics(x, inputs):
    # x: (neurons + 2, ), first two dims are readouts
    h = x[2:]
    h = h + dt*continuous_network_dynamics(h, inputs)
    y = phi(h).dot(C)
    x_new = np.concatenate((y, h))
    return x_new, x_new


# Combine them
def discrete_dynamics(x, inputs):
    """
    x: [y, h, q] of size 2+N+4 = N+6
    inputs: size (N, )
    """
    N = inputs.shape[0]
    network_states = discrete_network_dynamics(x[:N+2], inputs)[0]
    y, h = network_states[:2], network_states[2:]
    arm_states = arm_model.discrete_dynamics(x[N+2:], y)[0]
    x_new = np.concatenate((network_states, arm_states))
    return x_new, x_new 

def rollout(x0, u_trj):
    """
    x0: init states [y0, h0, q0], size (N+6, )
    u_trj: network inputs, size (N, )
    """
    N = u_trj.shape[1]
    _, x_trj = scan(discrete_dynamics, x0, u_trj)
    y, h, q = x_trj[:,:2], x_trj[:,2:N+2], x_trj[:,N+2:]
    return y, h, q

rollout_jit = jit(rollout)
rollout_batch = jit(vmap(rollout, in_axes=(0, 0)))