import equinox as eq
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Callable


"""
JAX-based workflow is slightly different.
Implement a curve that inherits from LearnableCurve (so, implement position, velocity, and acceleration).
LearnableCurve inherits from eq.Module, so it is a dataclass and satisfies a pytree.
By default optimize() trains every inexact array leaf. Override trainable_filter_spec(), or pass a
custom filter_spec to optimize(), to exclude specific leaves from optimization.
"""
class LearnableCurve(eq.Module):
    t0: float
    tf: float

    @abstractmethod
    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        pass

    def trainable_filter_spec(self):
        return jax.tree_util.tree_map(eq.is_inexact_array, self)

    @eq.filter_jit
    def curvature(self, t: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        v = self.velocity(t)
        a = self.acceleration(t)

        speed = jnp.linalg.norm(v, axis=-1)
        cross_mag = jnp.linalg.norm(jnp.cross(v, a, axis=-1), axis=-1)
        denom = speed ** 3

        return cross_mag / (denom + eps)

    @eq.filter_jit
    def max_curvature(self, samples = 4096):
        return jax.vmap(self.curvature)( jnp.linspace(self.t0, self.tf, samples) ).max()

    @eq.filter_jit
    def soft_max_curvature(self, key, samples = 4096, strength = 32):
        k = jax.vmap(self.curvature)( jax.random.uniform(key, shape=(samples,), minval=self.t0, maxval=self.tf) )
        return jax.scipy.special.logsumexp(k * strength, axis=0) / strength

    @eq.filter_jit
    def arclength(self, samples = 4096):
        t = jnp.linspace(self.t0, self.tf, samples)
        v = jax.vmap(self.velocity)(t)
        speed = jnp.linalg.norm(v, axis=-1)
        dt = (self.tf - self.t0) / (samples - 1)
        return jnp.sum((speed[:-1] + speed[1:]) * (0.5 * dt))

    @eq.filter_jit
    def _pi_theta_weight(self, order, theta):
        theta = jnp.asarray(theta)
        sign = jnp.where((order - theta - 1) % 2 == 0, 1.0, -1.0)
        logabs = (
            jax.scipy.special.gammaln(theta + 1.0)
            + jax.scipy.special.gammaln(order - theta)
            - jax.scipy.special.gammaln(order + 1.0)
        )
        return sign * jnp.exp(logabs)

    @eq.filter_jit
    def _pi_beta(self, velocities):
        complex_dtype = jnp.result_type(velocities.dtype, jnp.complex64)
        velocities = velocities.astype(complex_dtype)

        alpha0 = jnp.zeros(velocities.shape[0], dtype=complex_dtype)
        beta0 = velocities[:, 0, :]

        def body(carry, T):
            alpha_prev, beta_prev = carry
            alpha = jnp.sum(T * beta_prev, axis=-1)
            beta = alpha_prev[:, None] * T + 1j * jnp.cross(T, beta_prev, axis=-1)
            return (alpha, beta), None

        tail = jnp.swapaxes(velocities[:, 1:, :], 0, 1)
        (_, beta_n), _ = jax.lax.scan(body, (alpha0, beta0), tail)
        return beta_n

    @eq.filter_jit
    def magnus(self, order, key, samples = 4096):
        volume = (self.tf - self.t0) ** order
        sample_times = jax.random.uniform(
            key,
            shape=(samples, order),
            minval=self.t0,
            maxval=self.tf,
        )
        velocities = jax.vmap(self.velocity)(sample_times.reshape(-1)).reshape(samples, order, 3)
        theta = jnp.sum(sample_times[:, 1:] >= sample_times[:, :-1], axis=-1)
        weight = self._pi_theta_weight(order, theta).astype(velocities.dtype)
        beta_n = self._pi_beta(velocities)

        return jnp.real((1.0j**(1-order)) * volume * jnp.mean(weight[:, None] * beta_n, axis=0))

    def plot_curvature(self, samples = 4096):
        t_samples = jnp.linspace(self.t0, self.tf, samples)
        k_values = jax.vmap(self.curvature)(t_samples)

        plt.plot( t_samples, k_values )
        plt.xlabel(f'$ t \\in [{self.t0}, {self.tf}] $')
        plt.ylabel(r'Curvature $\kappa(t)$')
        plt.show() 

    def plot_position(self, samples = 4096):
        t_samples = jnp.linspace(self.t0, self.tf, samples)
        r_values = jax.vmap(self.position)(t_samples)

        fig = plt.figure(0, (5, 5))
        ax = fig.add_subplot(projection='3d')

        ax.plot( r_values[:, 0], r_values[:, 1] ,r_values[:, 2])
        fig.show()
    
    def optimize(
        self,
        cost_fn: Callable[["LearnableCurve", jnp.ndarray], jnp.ndarray],
        key: jnp.ndarray,
        lr: float = 1e-3,
        steps: int = 1000,
        chunk_size: int = 100,
        optimizer: optax.GradientTransformation = None,
        filter_spec = None,
        callback=None,
    ):
        if optimizer is None: optimizer = optax.adam(lr)

        assert steps % chunk_size == 0, "steps must be divisible by chunk_size"

        if filter_spec is None:
            filter_spec = self.trainable_filter_spec()

        params, static = eq.partition(self, filter_spec)
        opt_state = optimizer.init(params)

        @eq.filter_jit
        def train_chunk(params, opt_state, key):

            def body_fn(carry, _):
                params, opt_state, key = carry
                key, subkey = jax.random.split(key)

                def loss_fn(params, key):
                    model = eq.combine(params, static)
                    return cost_fn(model, key)

                loss, grads = eq.filter_value_and_grad(loss_fn)(params, subkey)

                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                return (params, opt_state, key), loss

            (params, opt_state, key), losses = jax.lax.scan(
                body_fn,
                (params, opt_state, key),
                None,
                length=chunk_size,
            )

            return params, opt_state, key, losses

        step_counter = 0

        for _ in range(steps // chunk_size):

            params, opt_state, key, losses = train_chunk(
                params, opt_state, key
            )

            if callback is not None:
                model = eq.combine(params, static)
                for loss in losses:
                    callback(step_counter, float(loss), model)
                    step_counter += 1

        return eq.combine(params, static)


class ArclenParameterize(LearnableCurve):
    base: LearnableCurve
    resolution: int
    newton_steps: int

    def __init__(self, base: LearnableCurve, resolution: int = 4096, newton_steps: int = 8):
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        if newton_steps < 1:
            raise ValueError("newton_steps must be >= 1")
        self.base = base
        self.resolution = int(resolution)
        self.newton_steps = int(newton_steps)
        self.t0 = 0.0
        self.tf = 1.0

    def _speed(self, t):
        return jnp.linalg.norm(self.base.velocity(t), axis=-1)

    def _arclength_to(self, t):
        t0 = self.base.t0
        tf = self.base.tf
        t = jnp.clip(jnp.asarray(t), t0, tf)

        x = jnp.linspace(0.0, 1.0, self.resolution)
        tau = t0 + (t - t0) * x
        speed = jax.vmap(self._speed)(tau)

        dx = 1.0 / (self.resolution - 1)
        return (t - t0) * jnp.sum((speed[:-1] + speed[1:]) * (0.5 * dx))

    def _total_arclength(self):
        return self._arclength_to(self.base.tf)

    def _invert(self, u):
        u = jnp.clip(jnp.asarray(u), self.t0, self.tf)
        base_t0 = self.base.t0
        base_tf = self.base.tf
        L = self._total_arclength()
        s_target = u * L

        def body(_, t):
            s = self._arclength_to(t)
            speed = self._speed(t)
            t = t - (s - s_target) / (speed + 1e-8)
            return jnp.clip(t, base_t0, base_tf)

        t_init = base_t0 + u * (base_tf - base_t0)
        t = jax.lax.fori_loop(0, self.newton_steps, body, t_init)
        return t, L

    @eq.filter_jit
    def arclength(self, samples = 4096):
        return self.base.arclength(samples)

    @eq.filter_jit
    def position(self, u):
        t, _ = self._invert(u)
        return self.base.position(t)

    @eq.filter_jit
    def velocity(self, u):
        t, L = self._invert(u)
        v = self.base.velocity(t)
        speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return L * v / (speed + 1e-8)

    @eq.filter_jit
    def acceleration(self, u):
        t, L = self._invert(u)
        v = self.base.velocity(t)
        a = self.base.acceleration(t)

        speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
        v_hat = v / (speed + 1e-8)
        proj = jnp.sum(a * v_hat, axis=-1, keepdims=True) * v_hat
        a_perp = a - proj

        return (L ** 2) * a_perp / ((speed + 1e-8) ** 2)
    
