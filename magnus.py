import jax
import jax.numpy as jnp
import time
import optax

from jax.scipy.special import bernoulli, factorial
from quadax import cumulative_simpson

from optspacecurve import FreeBarqCurve
import barqtools

from settings import settings

import numpy as np
import qutip as qt

def make_system_H(alpha=1.0, beta=0.7):
    return alpha * qt.sigmaz() + beta * qt.sigmax()

def unitary_from_piecewise_omega(H_sys, omega, dt):
    """
    omega: array of length N (piecewise constant amplitudes)
    dt: time step
    Returns:
        U (Qobj): joint unitary on system⊗ancilla for the interaction-picture dynamics
    """
    N = len(omega)
    tlist = np.arange(N+1) * dt  # N intervals, N+1 points

    # Precompute phi at the grid points (left Riemann sum for simplicity)
    phi = np.zeros(N+1)
    for k in range(N):
        phi[k+1] = phi[k] + omega[k] * dt

    # Operators
    Hy = qt.tensor(H_sys, qt.sigmay())
    Hz = qt.tensor(H_sys, qt.sigmaz())

    # Build coefficient arrays on the same grid expected by mesolve for "array coefficients"
    # We'll use values on tlist (length N+1). QuTiP will interpolate between them.
    cz = np.cos(2.0 * phi)
    cy = np.sin(2.0 * phi)

    H_td = [
        [Hz, cz],
        [Hy, cy],
    ]

    # Propagate unitary by evolving basis states (dimension d) or use propagator
    U = qt.propagator(H_td, tlist[-1], tlist=tlist)  # Qobj unitary on joint space
    return U

def kraus_block(U_joint, sys_dim):
    """
    Extract K0 = <0|U|0> acting on system, assuming ancilla is 2-dim and basis |0>,|1>.
    """
    # Joint dims assumed [sys, anc]
    # Build |0><0| on ancilla and take partial matrix elements
    b0 = qt.basis(2, 0)
    P00 = b0 * b0.dag()
    # (I⊗<0|) U (I⊗|0|) implemented via tensor identities:
    I_sys = qt.qeye(sys_dim)
    K0 = (qt.tensor(I_sys, b0.dag()) * U_joint * qt.tensor(I_sys, b0)).tidyup()
    return K0


def magnus_unitary_truncated(H_sys: qt.Qobj, B_list, n=None, *, sign=-1):
    """
    Build the n-th order truncated Magnus unitary:
        U^(n) = exp( sign * 1j * sum_{k=1..n} (H_sys^k ⊗ B_k) )

    Parameters
    ----------
    H_sys : qutip.Qobj
        Hermitian system operator (acts on system Hilbert space).
    B_list : list[qutip.Qobj]
        List of ancilla operators [B1, B2, ..., Bm] corresponding to Magnus terms.
        Each Bk acts only on ancilla Hilbert space.
    n : int | None
        Truncation order. If None, uses all terms in B_list.
    sign : int
        Use sign=-1 for Schrodinger convention U=exp(-i * Omega_Magnus).
        Use sign=+1 if your Pi_k were defined with the opposite sign.

    Returns
    -------
    U_trunc : qutip.Qobj
        Unitary on the joint space system⊗ancilla.
    Omega_trunc : qutip.Qobj
        The truncated Magnus generator sum_k H^k ⊗ B_k (useful for diagnostics).
    """
    if n is None:
        n = len(B_list)
    if n < 0 or n > len(B_list):
        raise ValueError(f"n must be between 0 and len(B_list)={len(B_list)}")

    if n == 0:
        # identity on joint space
        anc_dim = B_list[0].shape[0] if len(B_list) > 0 else 1
        return qt.tensor(qt.qeye(H_sys.shape[0]), qt.qeye(anc_dim)), 0

    # infer ancilla identity from B1 dims
    anc_I = qt.qeye(B_list[0].shape[0])

    Omega = 0 * qt.tensor(qt.qeye(H_sys.shape[0]), anc_I)  # zero operator on joint space
    Hpow = qt.qeye(H_sys.shape[0])  # H^0
    for k in range(1, n + 1):
        Hpow = Hpow * H_sys            # now H^k
        Bk = B_list[k - 1]
        Omega = Omega + qt.tensor(Hpow, Bk)

    U = (sign * 1j * Omega).expm()
    return U, Omega

def vector_int_cumul(vector, x_values):

    """
    Provides the cumulative integral for a vector function.
    Assumes that the vectors are arranged by rows: n_samples x 3.
    """

    return cumulative_simpson(y=vector, x=x_values, axis=0,
                                              initial=0)

def _ordered_partitions(total, parts):

    """
    Returns ordered partitions of ``total`` into ``parts`` positive integers.
    """

    if parts == 1:
        return [(total,)]

    partitions = []

    for first in range(1, total - parts + 2):
        for remaining in _ordered_partitions(total - first, parts - 1):
            partitions.append((first, *remaining))

    return partitions


def _nested_cross(vectors):

    """
    Left-associated nested cross product for a sequence of vectors.
    """

    result = vectors[0]

    for vector in vectors[1:]:
        result = jnp.cross(result, vector)

    return result
    

# def generate_magnus_integrand(frenet_dict, n):
#     if n < 1:
#         raise ValueError("Magnus Index must be at least 1")

#     if n == 1:
#         def 
    

def calculate_magnus_term(frenet_dict, n, magnus_terms=None):

    r"""
    Calculates the n-th Magnus term using the recurrence

    .. math::
        \vec{\Pi}_n(t) = (2i)^{n-1}\sum_{j=1}^{n-1}\frac{B_j}{j!}
        \int_0^t \left[\sum_{i_1\ldotsi_j=n-1}
        \vec{\Pi}_{i_1}(\tau)\times\cdots\times\vec{\Pi}_{i_j}(\tau)
        \times\vec{T}(\tau)\right]d\tau,

    with

    .. math::
        \vec{\Pi}_1(t) = \int_0^t \vec{T}(\tau) d\tau.

    Args:
        frenet_dict (dict): Frenet dictionary with keys ``frame`` and
            ``x_values``.
        n (int): Magnus term index, starting from one.
        magnus_terms (dict|None): Optional cache for already computed terms.

    Returns:
        jax.Array: Numerical samples of :math:`\vec{\Pi}_n(t)` for every point
        in ``frenet_dict['x_values']``.
    """

    if n < 1:
        raise ValueError('The Magnus index n must satisfy n >= 1.')

    if magnus_terms is None:
        magnus_terms = {}

    if n in magnus_terms:
        return magnus_terms[n], magnus_terms

    tangent = frenet_dict['frame'][:, 0, :]
    times = frenet_dict['x_values']

    if 1 not in magnus_terms:
        magnus_terms[1] = vector_int_cumul(tangent, times)

    if n == 1:
        return magnus_terms[1], magnus_terms

    bernoulli_nums = bernoulli(n - 1)
    integrand = jnp.zeros_like(magnus_terms[1])

    for j in range(1, n):
        if j > 1 and j%2 == 1:
            continue
            
        bj = bernoulli_nums[j]


        nested_sum = jnp.zeros_like(magnus_terms[1])

        for part in _ordered_partitions(n - 1, j):
            pi_vectors = [calculate_magnus_term(frenet_dict, idx, magnus_terms)[0]
                          for idx in part]
            nested_sum += _nested_cross([*pi_vectors, tangent])

        integrand += (2**j)*(bj/factorial(j))*nested_sum

    magnus_terms[n] = vector_int_cumul(integrand, times)

    return magnus_terms[n], magnus_terms


@jax.jit
def magnus_loss(frenet_dict, target_values, loss_weights = None):

    if loss_weights is None:
        loss_weights = jnp.ones(target_values.shape[0])
    
    start = time.time()
    magnus_vals = calculate_magnus_term(frenet_dict, target_values.shape[0])[1]
    magnus_terms = jnp.array([magnus_vals[i+1][-1] for i in range(target_values.shape[0])])

    dist = jnp.dot(loss_weights,jnp.sum((magnus_terms - target_values) ** 2, axis = 1))

    
    return dist

#from optspacecurve import safe_scale, free_barq_fun
## def safe_scale(vec):
## def free_barq_fun(params, end_point):

#if __name__ == "__main__":

    #H_sys = make_system_H()
    #sys_dim = H_sys.shape[0]

    #N = 200
    #dt = 0.01
    #omega = 2.0 * np.ones(N)  # initial guess

    #def curve_from_control(omega, N, dt):
        ##Curve looks like r(t) dot sigma = int U_I sigma_z U_I where
        ##U_I = exp(int Omega(t))
        ##We 
        ##The integrand works out to cos(2 phi(t)) sigma_z + sin(2 phi) sigma_y where
        ## phi = int Omega(t)
        #phi = np.zeros(N+1)
        #for k in range(N):
            #phi[k+1] = phi[k] + omega[k] * dt

        #z_coef = np.cos(2*phi)

        #y_coef = np.sin(2*phi)

        #curve_cum = np.zeros(N+1)
        #for k in range(N):
            #curve_cum[k+1] += 
            

        
        

        

    #U = unitary_from_piecewise_omega(H_sys, omega, dt)
    #K0 = kraus_block(U, sys_dim)

    #faux_frenet = {}

    #faux_frenet['x_values'] = np.arange(0,dt*N, dt)

    #faux_frenet['frame']
    

    #Umagnus_unitary_truncated(H_sys, B_list, n=None, *, sign=-1)
    
    ## params = optax.apply_updates(params, updates)
    
    ## initial = barqcurve.params['free_points']
    
    ## optimizer = optax.adamw(learning_rate=1e-3)
    
    ## grad_mask = jax.tree.map(lambda x: jnp.ones_like(x), barqcurve.params)
    
    ## for key in grad_mask['pgf_params']:
    ##     grad_mask['pgf_params'][key] = jnp.array(0.0)  # freeze this
    
    ## barqcurve.optimize_fast(
    ##     optimizer=optimizer,
    ##     max_iter=5,
    ##     grad_mask=grad_mask,
    ##     record_every=10
    ## )
    
    ## print(barqcurve.params['free_points'])
    ## print(f"Total Movement: {jnp.sum((barqcurve.params['free_points'] - initial)**2)}")