import jax
from jax import numpy as np
from jax.lax import scan
from jax import vmap, jit
from jax import jacfwd, jacrev
import numpy as onp

import pickle
import time

import network_and_arm
import arm_model

# Global Variable(s)
#TODO: Would be helpful if these could be defined in notebook
# Delay period from Schimel et al. 2024
DELAY = 300 #ms
# Move period from Schimel et al. 2024
T = 900

def init_radial_task(start_pos=np.array([0.0, 0.4]), radius=0.12):
    """
    Initialize the arm model in a radial task. Targets are placed in a circle
    around the starting position of the hand.
    Args:
        start_pos (np.ndarray): Starting position of the hand [x, y]
        radius (float): Radius of the radial task
    Returns:
        init_thetas (np.ndarray): Initial joint angles [theta1, theta2]
        target_angles (np.ndarray): Target joint angles [theta1, theta2]
        targets (np.ndarray): Target positions [x, y]
    """
    # Set starting positions
    x0, y0 = start_pos

    # Get target locations
    target_angles = (2*np.pi/360)* np.array(
        [0., 45., 90., 135., 180., 225., 270., 315.]
    )
    target_x = x0 + (np.cos(target_angles)*radius)
    target_y = y0 + (np.sin(target_angles)*radius)
    targets = np.concat([target_x[:, None], target_y[:, None]], axis=1) #m

    # Get initial angles from starting position
    theta1, theta2 = arm_model.get_angles(start_pos)
    init_thetas = np.vstack([theta1, theta2]).squeeze()

    # Store arm angles for targets
    target_t1, target_t2 = jax.vmap(arm_model.get_angles)(targets)
    target_angles = np.hstack([target_t1[:, None], target_t2[:, None]])
    
    return init_thetas, target_angles, targets

# NOTE: if changing start pos in notebook, START_ANGLE will be wrong
_, START_ANGLE, _ = init_radial_task()

# Cost functions
# NOTE: Cost from SCHIMEL et al.
def cost_stage(x, u, t, target, lmbda):
    """
    \|state - target\|^2 + lmbda \|u\|^2
    x: (n_states,) angles 
    u: (n_controls, )
    target: (n_states, ) -- To Do
    lmbda: float > 0, penalty on cost
    """
    ALPHA_NULL, ALPHA_EFFORT = lmbda
    # Create weights to mimic integral bounds while keeping differentiability
    move_weight = (jax.nn.relu(t-DELAY)**2/T**2)
    prep_weight = ALPHA_NULL * (jax.nn.relu(DELAY-t)**2/DELAY**2)
    
    # Cost during movement phase (weighted by move_weight)
    movement_cost = move_weight * np.sum((x[-4:-2] - target)**2)
    
    # Cost during preparation phase (weighted by prep_weight)
    # Interesting, prep_weight > 0.01 causes NaNs (only checked log factor)
    prep_cost = prep_weight * (np.sum((x[-4:-2] - START_ANGLE)**2) + 
                               np.sum(x[-2:]**2) + np.sum(x[:2]**2))
    
    # Control cost remains the same
    control_cost = ALPHA_EFFORT * np.sum(u**2)
    
    return movement_cost + prep_cost + control_cost

def cost_final(x, target):
    """
    \|state - target\|^2 
    """
    return np.sum((x[-4:-2] - target)**2)

# Computes cost over trajectory of ln. (time steps, n_states)
cost_stage_trj = vmap(cost_stage, in_axes=(0,0,0,0,None))
# Cost of multiple trajectories: (batch size, time steps, n_states)
cost_stage_batch = vmap(cost_stage_trj, in_axes=(0,0,0,0,None))

def cost_trj(x_trj, u_trj, target_trj, lmbda):
    """
    \sum_t \|state - target\|^2 + lmbda \|u\|^2
    """
    t = np.arange(x_trj.shape[0]-1)
    c_stage = cost_stage_trj(x_trj[:-1], u_trj[:-1], t, target_trj[:-1], lmbda) 
    c_final = cost_final(x_trj[-1], target_trj[-1])
    return c_stage + c_final

cost_trj_batch = vmap(cost_trj, (0,0,0,None))


#  Gradients
def cost_stage_grads(x, u, t, target, lmbda):
    """
    x: (n_states, )
    u: (n_controls,)
    target: (n_states, )
    lmbda: penalty on controls 
    """
    
    dL = jacrev(cost_stage, (0,1)) #l_x, l_u
    d2L = jacfwd(dL, (0,1)) # l_xx etc
    
    l_x, l_u = dL(x, u, t, target, lmbda)
    d2Ldx, d2Ldu = d2L(x, u, t, target, lmbda)
    l_xx, l_xu = d2Ldx
    l_ux, l_uu = d2Ldu
    
    return l_x, l_u, l_xx, l_ux, l_uu

# Accepts (batch size, n_states) etc.
cost_stage_grads_batch = vmap(cost_stage_grads, in_axes=(0,0,0,0,None))

def cost_final_grads(x, target):
    """
    x: (n_states, )
    target: (n_states, )
    """
    dL = jacrev(cost_final) #l_x, l_u
    d2L = jacfwd(dL) # l_xx etc
    
    l_x = dL(x, target)
    l_xx = d2L(x, target)
    
    return l_x, l_xx

cost_final_grad_batch = vmap(cost_final_grads, in_axes=(0,0))

def dynamics_grads(x, u):
    """
    f: discrete dynamics x[t+1] = f(x[t], u[t])
    """
    def f(x,u):
        # Grab first output 
        return network_and_arm.discrete_dynamics(x,u)[0]
    
    f_x, f_u = jacfwd(f, (0,1))(x,u)
    return f_x, f_u
  
dynamics_grads_batch = vmap(dynamics_grads, (None,0,0))


### Helpers for LQR approximation ### 

def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    """
    Assemble coefficients for quadratic approximation of value fn
    """
    Q_x = l_x.T + V_x.T @ f_x 
    Q_u = l_u.T + V_x.T @ f_u 
    Q_xx = l_xx + f_x.T @ V_xx @ f_x #
    Q_ux = l_ux + f_u.T @ V_xx @ f_x #
    Q_uu = l_uu + f_u.T @ V_xx @ f_u #
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


def gains(Q_uu, Q_u, Q_ux):
    """
    Feedback control law u* = k + Kx*
    """
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = np.zeros(Q_u.shape) - Q_uu_inv @ Q_u
    K = np.zeros(Q_ux.shape) - Q_uu_inv @ Q_ux
    return k, K

def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    """
    Quadratic approximation of value function
    """
    V_x  = Q_x.T + Q_u.T @ K + k @ Q_ux + k.T @ Q_uu @ K
    V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K
    return V_x, V_xx

def expected_cost_reduction(Q_u, Q_uu, k):
    """
    Assuming approximations are true
    """
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


# Forward pass
def discrete_dynamics_affine(xs, inputs):
    """
    Wrapper around arm dynamics fun that pre-computes
    control law
    """
    ut, xt_new = xs[:200], xs[200:]
    xt, ut, kt, Kt = inputs
    ut_new = ut + kt + Kt@(xt_new - xt)
    xt_new2 = network_and_arm.discrete_dynamics(xt_new, ut_new)[0]
    res = np.concatenate((ut_new, xt_new2))
    return res, res 


def forward_pass_scan(x_trj, u_trj, k_trj, K_trj):
    """
    Simulate the system using control law around (x_trj, u_trj)
    defined by k_trj, K_trj
    """
    inputs = (x_trj, u_trj, k_trj, K_trj)
    init = np.concatenate((np.zeros_like(u_trj[0]), x_trj[0]))
    states  = scan(discrete_dynamics_affine, init, inputs)[1]
    u_trj_new, x_trj_new = states[:,:200], states[:-1,200:]
    #print(x_trj[0].shape, x_trj_new.shape)
    x_trj_new = np.concatenate((x_trj[0][None], x_trj_new), axis=0)
    return u_trj_new, x_trj_new

forward_pass_jit = jit(forward_pass_scan)
# Batch over x, u, and feedback
forward_pass_batch = jit(vmap(forward_pass_scan, (0,0,0,0))) 

# Backward pass
def step_back_scan(state, inputs, regu, lmbda):
    """
    One step of Bellman iteration, backward in time
    """
    x_t, u_t, t, target_t = inputs
    k, K, V_x, V_xx = state
    l_x, l_u, l_xx, l_ux, l_uu = cost_stage_grads(x_t, u_t, t, target_t, lmbda)
    f_x, f_u = dynamics_grads(x_t, u_t)
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    new_state = (k, K, V_x, V_xx)
    return new_state, new_state

def backward_pass_scan(x_trj, u_trj, target_trj, regu, lmbda):
    """
    Bellman iteration over entire trajectory
    """
    n_x, n_u = x_trj.shape[1], u_trj.shape[1]
    k, K = np.zeros((n_u, )), np.zeros((n_u, n_x))
    l_final_x, l_final_xx = cost_final_grads(x_trj[-1], target_trj[-1])
    V_x = l_final_x
    V_xx = l_final_xx
    # Wrap initial state and inputs for use in scan
    init = (k, K, V_x, V_xx)
    ts = np.arange(x_trj.shape[0])
    xs = (x_trj, u_trj, ts, target_trj)
    # Loop --- backward in time
    step_fn = lambda state, inputs: step_back_scan(state, inputs, regu, lmbda)
    _, state = scan(step_fn, init, xs, reverse=True)
    k_trj, K_trj, _, _ = state
    return k_trj, K_trj

backward_pass_jit = jit(backward_pass_scan)


def run_ilqr(x0, target_trj, u_trj = None, max_iter=10, regu_init=10, lmbda=1e-1):
    # Main loop
    # First forward rollout
    if u_trj is None:
        N = target_trj.shape[0]
        n_u = 200
        u_trj = onp.random.normal(size=(N, n_u)) * 0.0001
    y_trj, h_trj, q_trj  = network_and_arm.rollout(x0, u_trj)
    x_trj = np.concatenate((y_trj, h_trj, q_trj),1)
    total_cost = cost_trj(x_trj, u_trj, target_trj, lmbda).sum()
    regu = regu_init
    
    cost_trace = [total_cost]
    x_trace = [x_trj]
    u_trace = [u_trj]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj = backward_pass_jit(x_trj, u_trj, target_trj, regu, lmbda)
        u_trj_new, x_trj_new = forward_pass_jit(x_trj, u_trj, k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new, target_trj, lmbda).sum()
        
        cost_redu = cost_trace[-1] - total_cost
        cost_trace.append(total_cost)
        # Still need to update, start from current guess
        x_trj = x_trj_new
        u_trj = u_trj_new

        # Add to trace
        x_trace.append(x_trj)
        u_trace.append(u_trj)
        
        # if it%1 == 0:
        #    print(f"Iteration {it}: Cost: {total_cost}, Reduction: {cost_redu}")
    
    # Get index of lowest cost
    min_cost_idx = np.hstack(cost_trace).argmin()
    # Get best trajectoies
    best_x_trj = np.array(x_trace)[min_cost_idx]
    best_u_trj = np.array(u_trace)[min_cost_idx]
    
    return best_x_trj, best_u_trj, np.array(cost_trace)

# To do: use scan and jit?. At least vmap the backward passes etc.
run_ilqr_batch = vmap(run_ilqr, (0, 0, 0, None, None, None))