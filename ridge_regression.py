import jax
import equinox as eq
import jax.numpy as jnp
import numpy as np
from typing import Callable

import matplotlib.pyplot as plt

@eq.filter_jit
def make_y(fn: Callable[[jnp.ndarray], jnp.ndarray], degree: int, samples: int = 128, dtype=None):
    nodes, weights = np.polynomial.legendre.leggauss(samples)
    x = jnp.asarray(np.r_[(nodes - 1) / 2, (nodes + 1) / 2], dtype=dtype)
    weights = jnp.asarray(np.r_[weights / 2, weights / 2], dtype=dtype)
    values = jax.vmap(lambda t: fn(t) + jnp.zeros_like(t))(x)
    return (x[:, None] ** jnp.arange(degree + 1)).T @ (weights * values)

@eq.filter_jit
def make_P(degree: int, dtype=None):
    i = jnp.arange(degree + 1)
    powers = i[:, None] + i[None, :]
    return jnp.asarray(jnp.where(powers % 2, 0, 2 / (powers + 1)), dtype=dtype)

@eq.filter_jit
def generate_coefficients(fn: Callable[[jnp.ndarray], jnp.ndarray], lam_seq: jnp.ndarray, degree: int, y: jnp.ndarray | None = None, P: jnp.ndarray | None = None):
    if y is None: y = make_y(fn, degree)
    P = make_P(degree) if P is None else jnp.asarray(P)
    return jnp.linalg.solve(P + jnp.diag(jnp.asarray(lam_seq)), jnp.asarray(y))

def plot_approximation(
        target_fn: Callable[[jnp.ndarray], jnp.ndarray], 
        poly_coefs: jnp.ndarray, 
        samples = 4096,
        target_fn_pre_op: Callable[[jnp.ndarray], jnp.ndarray] | None = None, 
        poly_post_op: Callable[[jnp.ndarray], jnp.ndarray] | None = None, 
    ):
    x_values = jnp.linspace(-1, 1, samples)
    y_values = jnp.polyval(poly_coefs[::-1], x_values)

    plt.plot(x_values, target_fn(x_values), "k--", label="step")
    plt.plot(x_values, y_values, label="p")

    if target_fn_pre_op is not None: plt.plot(x_values, target_fn_pre_op(target_fn(x_values)), "--", label="arcos(step)")
    if poly_post_op is not None: plt.plot(x_values, poly_post_op(y_values), label="cos(p)")

    plt.ylim(-0.25, 1.85)
    plt.legend()
    plt.show()

@eq.filter_jit
def find_max_abs_deriv(coefs: jnp.ndarray):
    first_deriv = jnp.polyder(jnp.asarray(coefs)[::-1])
    second_deriv = jnp.polyder(first_deriv)
    endpoints = jnp.max(jnp.abs(jnp.polyval(first_deriv, jnp.array([-1., 1.]))))
    if second_deriv.size <= 1: return endpoints

    roots = jax.lax.stop_gradient(jnp.roots(second_deriv, strip_zeros=False))
    valid = (jnp.abs(roots.imag) < 1e-10) & (roots.real >= -1) & (roots.real <= 1)
    points = jnp.where(valid, roots.real, 0)
    interior = jnp.where(valid, jnp.abs(jnp.polyval(first_deriv, points)), 0)
    return jnp.maximum(endpoints, jnp.max(interior))

@jax.jit
def constant_weights(n, l0=1e-3):
    return l0 * jnp.ones(n + 1)

def exp_growth_weights(n, l0=1e-6, alpha=0.25):
    return jnp.asarray([l0 * jnp.exp(alpha * np.asarray(i)) for i in range(n + 1) ])

def polynomial_growth_weights(n, l0=1e-8, power=4):
    return jnp.asarray([l0 * (jnp.asarray(i) + 1) ** power for i in range(n + 1) ])
