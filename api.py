### Main module for the local optimization scheme
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial 
from jax import vmap, jit 

np.random.seed(0)

@partial(jit, static_argnums=(1,))
def moving_window(beta, delta_omega):
    """
    Returns a matrix of size (size+1, size) with each element ij being an
    addition of the terms a_i and a_j.
    
    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        size (int): resolution of the matrices
    returns:
        array[float]: Hankel matrix representing the pump
    """
    size = (len(beta) + 1)//2
    starts = jnp.arange(len(beta) - size + 1)
    return delta_omega*vmap(lambda start: jax.lax.dynamic_slice(beta, (start,), (size,)))(starts)

def build_propagator_plus(beta, omega, z):
    """
    Gives the positive propagator

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
    returns:
        array[float]: propagator associated with sum of PMF and pump
    """
    delta_omega = np.abs(omega[1] - omega[0])
    beta_mat = moving_window(beta, delta_omega)
    delta_k = (1.j)*jnp.diag(omega)
    Q = delta_k + beta_mat
    W_plus = jscipy.linalg.expm(Q*z)
    return W_plus

def build_propagator_minus(beta, omega, z):
    """
    Gives the negative propagator

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
    returns:
        array[float]: propagator associated with difference of PMF and pump
    """
    delta_omega = np.abs(omega[1] - omega[0])
    beta_mat = moving_window(beta, delta_omega)
    delta_k = (1.j)*jnp.diag(omega)
    Q = delta_k - beta_mat
    W_minus = jscipy.linalg.expm(Q*z)
    return W_minus

def build_J_matrix(beta, omega, z):
    """
    Gives the quadratic of positive propagator minus negative propagator

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
    returns:
        array[float]: matrix associated with mean photon number
    """
    W_plus = build_propagator_plus(beta, omega, z)
    W_minus = build_propagator_minus(beta, omega, z)
    J = 0.25*(W_plus@W_plus.conj().T + W_minus@W_minus.conj().T - 2*jnp.eye(len(omega)))
    return J

def get_observables(beta, omega, z):
    """
    Gives the mean photon number and the Schmidt number of the problem

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
    returns:
        (float, float): the mean photon number per pulse and the Schmidt number
    """
    J = build_J_matrix(beta, omega, z)
    photon_nbr = jnp.real(jnp.trace(J))
    obj_func = jnp.real((jnp.trace(J)**2)/jnp.trace(J.conj().T@J) - 1)
    return photon_nbr, obj_func

def get_penalty_photon_numb(beta, omega, z, n):
    """
    Gives the penalty on mean number of photon pair per pulse

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
        n (float): targeted mean number of photon pair per pulse
    returns:
        (float): l2 norm on the mean photon number from propagators and targeted
    """
    photon_nbr, _ = get_observables(beta, omega, z)
    return (photon_nbr - n)**2
    
def get_penalty_pump_shape(beta, omega):
    """
    Gives the penalty on variance of the pump taking into account the normalized pump behaves
    like a propability distribution

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
    returns:
        [float]: penalty on pump variance
    """
    N_omega = len(omega)
    omega_scale = jnp.diag(jnp.linspace(omega[0], omega[-1], 2*N_omega - 1))
    omega_scale = omega_scale@omega_scale
    problem_cov = (beta.conj().T@omega_scale@beta)/(beta.conj().T@beta)
    return .2*problem_cov

def problem(beta, omega, z, n, k):
    """
    Gives the function to minimize taking into account the mean photon pair and the variance of pump

    Args: 
        beta (array[float]): problem parameters. Length is 2*N_omega - 1
        omega (array[float]): discretized frequency domain of length N_omega
        z (float): length of the waveguide
        n (float): targeted mean number of photon pair per pulse
        k (int): the penalty value
    returns:
        (float): the value of the penalized objective function
    """
    _, obj_f = get_observables(beta, omega, z)
    penalty_photon_nbr = get_penalty_photon_numb(beta, omega, z, n)
    penalty_pump_shape = get_penalty_pump_shape(beta, omega)
    return obj_f + k*(penalty_photon_nbr) + penalty_pump_shape