import equinox as eq
import jax
import jax.numpy as jnp
import optax
from abc import abstractmethod
from typing import Callable

class LearnableCurve(eq.Module):
    t0: float
    tf: float

    @abstractmethod
    def position(self, t: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def velocity(self, t: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def acceleration(self, t: jax.Array) -> jax.Array:
        pass

    @eq.filter_jit
    def curvature(self, t: jax.Array, eps: float = 1e-8) -> jax.Array:
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

    def optimize(
        self,
        cost_fn: Callable[["LearnableCurve", jax.Array], jax.Array],
        key: jax.Array,
        lr: float = 1e-3,
        steps: int = 1000,
        chunk_size: int = 100,
        optimizer: optax.GradientTransformation = None,
        callback=None,
    ):
        if optimizer is None: optimizer = optax.adam(lr)

        assert steps % chunk_size == 0, "steps must be divisible by chunk_size"

        params, static = eq.partition(self, eq.is_inexact_array)
        opt_state = optimizer.init(params)

        @eq.filter_jit
        def train_chunk(params, opt_state, key):

            def body_fn(carry, _):
                params, opt_state, key = carry
                key, subkey = jax.random.split(key)

                model = eq.combine(params, static)

                loss, grads = eq.filter_value_and_grad(cost_fn)(model, subkey)

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

    def __init__(self, base: LearnableCurve, resolution: int = 4096):
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        self.base = base
        self.resolution = int(resolution)
        self.t0 = 0.0
        self.tf = 1.0

    def _arclen_table(self):
        N = self.resolution
        t0 = self.base.t0
        tf = self.base.tf

        t = jnp.linspace(t0, tf, N)
        v = jax.vmap(self.base.velocity)(t)
        speed = jnp.linalg.norm(v, axis=-1)

        dt = (tf - t0) / (N - 1)
        ds = (speed[:-1] + speed[1:]) * (0.5 * dt)

        s = jnp.zeros_like(t)
        s = s.at[1:].set(jnp.cumsum(ds))

        L = s[-1]

        return t, s, L

    def _invert_with_table(self, u, t_samples, s_samples, L):
        u = jnp.asarray(u)
        u = jnp.clip(u, self.t0, self.tf)
        s_query = u * L
        idx = jnp.searchsorted(s_samples, s_query, side="right") - 1
        idx = jnp.clip(idx, 0, self.resolution - 2)
        s0 = s_samples[idx]
        s1 = s_samples[idx + 1]
        t0 = t_samples[idx]
        t1 = t_samples[idx + 1]
        inv = 1.0 / (s1 - s0 + 1e-12)
        w = (s_query - s0) * inv
        return t0 + w * (t1 - t0)

    def _invert(self, u):
        t_samples, s_samples, L = self._arclen_table()
        return self._invert_with_table(u, t_samples, s_samples, L)

    def _frozen_arclen_table(self):
        t_samples, s_samples, L = self._arclen_table()
        return (
            jax.lax.stop_gradient(t_samples),
            jax.lax.stop_gradient(s_samples),
            jax.lax.stop_gradient(L),
        )

    @eq.filter_jit
    def position(self, s):
        t_samples, s_samples, L = self._frozen_arclen_table()
        t = self._invert_with_table(s, t_samples, s_samples, L)
        return self.base.position(t)

    @eq.filter_jit
    def velocity(self, u):
        t_samples, s_samples, L = self._frozen_arclen_table()
        t = self._invert_with_table(u, t_samples, s_samples, L)

        v = self.base.velocity(t)
        speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return L * v / (speed + 1e-8)

    @eq.filter_jit
    def acceleration(self, u):
        t_samples, s_samples, L = self._frozen_arclen_table()
        t = self._invert_with_table(u, t_samples, s_samples, L)

        v = self.base.velocity(t)
        a = self.base.acceleration(t)

        speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
        v_hat = v / (speed + 1e-8)
        proj = jnp.sum(a * v_hat, axis=-1, keepdims=True) * v_hat
        a_perp = a - proj

        return (L ** 2) * a_perp / ((speed + 1e-8) ** 2)
