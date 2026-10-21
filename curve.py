import jax
import optax
import jax.numpy as jnp
import equinox as eq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from warnings import warn
from abc import abstractmethod
from typing import Callable, Self

import magnus.jax as magnus

"""
JAX-based workflow is slightly different.
Implement a curve that inherits from LearnableCurve (so, implement position, velocity, acceleration, and jerk).
You MUST implement position at the minimum. Velocity, acceleration, and jerk use JAX autodifferentiation as fallback if the implementations
are too complex or too trivial. A warning will be made at __init__.
If position accepts batched times, it should evaluate each time independently.
LearnableCurve inherits from eq.Module, so it is a dataclass and satisfies a pytree.
By default optimize() trains every inexact array leaf. Override trainable_filter_spec(), or pass a
custom filter_spec to optimize(), to exclude specific leaves from optimization.
"""
class LearnableCurve(eq.Module):
    t0: float
    tf: float

    def __init__(self, t0: float, tf: float):
        self.t0 = t0
        self.tf = tf

        cls = self.__class__
        if cls.velocity is LearnableCurve.velocity: warn(f"{cls.__name__}.velocity not implemented, using jax.jvp fallback")
        if cls.acceleration is LearnableCurve.acceleration: warn(f"{cls.__name__}.acceleration not implemented, using jax.jvp fallback")
        if cls.jerk is LearnableCurve.jerk: warn(f"{cls.__name__}.jerk not implemented, using jax.jvp fallback")

    @abstractmethod
    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        pass

    """
    Should be overloaded, but if not possible or non-trivial, JAX autodifferentiation is used in place
    """
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(self.position, (t,), (jnp.ones_like(t),))[1]

    """
    Should be overloaded, but if not possible or non-trivial, JAX autodifferentiation is used in place
    """
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(self.velocity, (t,), (jnp.ones_like(t),))[1]

    """
    Should be overloaded, but if not possible or non-trivial, JAX autodifferentiation is used in place
    """
    def jerk(self, t: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(self.acceleration, (t,), (jnp.ones_like(t),))[1]

    def trainable_filter_spec(self):
        return jax.tree_util.tree_map(eq.is_inexact_array, self)

    @eq.filter_jit
    def curvature(self, t: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        v = self.velocity(t)
        a = self.acceleration(t)
        eps = jnp.asarray(eps, dtype=jnp.result_type(v.dtype, a.dtype))

        speed_sq = jnp.sum(v * v, axis=-1)
        cross_sq = jnp.sum(jnp.cross(v, a, axis=-1) ** 2, axis=-1)

        speed = jnp.sqrt(speed_sq + eps * eps)
        cross_mag = jnp.sqrt(cross_sq + eps * eps) - eps

        return cross_mag / (speed ** 3)

    @eq.filter_jit
    def torsion(self, t: jnp.ndarray, eps = 1e-8) -> jnp.ndarray:
        v = self.velocity(t)
        a = self.acceleration(t)
        j = self.jerk(t)

        v_x_a = jnp.cross(v, a, axis=-1)

        numer = v_x_a.dot( j )
        denom = jnp.sum( v_x_a**2 ) + eps

        return numer / denom

    @eq.filter_jit
    def phi(self, t: jnp.ndarray, samples=4096) -> jnp.ndarray:
        time = jnp.linspace(self.t0, t, samples)
        torsion = jax.vmap( self.torsion )(time)
        speed = jax.vmap( lambda x: jnp.linalg.norm( self.velocity(x) ) )(time)

        d3r_x, d3r_y, _ = self.jerk(self.t0)
        azimuth_fix = jnp.atan2( -d3r_x, d3r_y )

        return jax.scipy.integrate.trapezoid( y=speed * torsion, x=time ) + azimuth_fix

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
        
        return jax.scipy.integrate.trapezoid(y=speed, x=t)

    @eq.filter_jit
    def magnus(self, n: int, samples = 4097):
        """
        Requires pulling in magnus-dp package: `https://github.com/s-cavalier/magnus-dp`
        Also drops the first term.
        """
        return magnus.compute_sc(n, self.velocity, self.t0, self.tf, samples, op="many", vectorized=False)[1:]

    @eq.filter_jit
    def tangent(self, t: float) -> jnp.ndarray:
        velocity = self.velocity(t)
        return velocity / jnp.linalg.norm(velocity)

    @eq.filter_jit
    def error_bound_control(self, samples=4096):
        ts = jnp.linspace(0.0, self.tf, samples)
        Ts = jax.vmap(self.tangent)(ts)

        G = Ts @ Ts.T

        vals = jnp.sqrt(jnp.clip(1.0 - G**2, 0.0, 1.0))

        vals = vals - jnp.eye(samples) * 2.0  

        return jnp.max(vals)

    @eq.filter_jit
    def error_bound(self, H_in: jnp.ndarray, degree: int, minimizer_samples=4096, arclen_samples=4096):
        DELTA_CONSTANT = 0.920075

        h_max = jnp.linalg.matrix_norm(H_in, ord=2)

        t = self.arclength(arclen_samples)   
        rho = DELTA_CONSTANT * h_max * t

        eta = self.error_bound_control()

        return (4.0 * eta / ((degree + 1) ** 2)) * (rho ** (degree + 1)) / (1.0 - rho)

    @eq.filter_jit
    def test_error_convergence(self, H_in: jnp.ndarray):
        DELTA_CONSTANT = 0.920075
        h_max = jnp.linalg.matrix_norm(H_in, ord=2)
        t = self.arclength()

        return DELTA_CONSTANT * h_max * t < 1


    # fancy plots

    def plot_curvature(self, samples = 4096):
        t_samples = jnp.linspace(self.t0, self.tf, samples)
        k_values = jax.vmap(self.curvature)(t_samples)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_samples, k_values, color="black", linewidth=2.2)
        ax.fill_between(t_samples, k_values, color="black", alpha=0.08)
        ax.set_xlabel(f'$ t \\in [{self.t0}, {self.tf}] $')
        ax.set_ylabel(r'Curvature (Pulse) $\kappa(t) = \Omega(t)$')
        ax.set_title(r"Curvature $\kappa(t)$ (also Pulse, $\Omega(t)$)")
        ax.grid(alpha=0.25, linewidth=0.8)
        fig.tight_layout()
        plt.show()

    def plot_torsion(self, samples = 1024):
        t_samples = jnp.linspace(self.t0, self.tf, samples)
        tau_values = jax.vmap(self.torsion)(t_samples)
        phi_values = jax.vmap(self.phi)(t_samples)

        fig, ax = plt.subplots(figsize=(7.5, 4.25))
        ax_phi = ax.twinx()

        tau_line = ax.plot(
            t_samples,
            tau_values,
            color="#0F766E",
            linewidth=2.2,
            label=r'$\tau(t)$',
        )[0]
        ax.fill_between(t_samples, tau_values, 0.0, color="#0F766E", alpha=0.12)
        phi_line = ax_phi.plot(
            t_samples,
            phi_values,
            color="#D97706",
            linewidth=2.2,
            label=r'$\phi(t) = \int_{t_0}^{t} \tau(s)\,ds$',
        )[0]

        ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle=":")
        ax.set_xlabel(f'$ t \\in [{self.t0}, {self.tf}] $')
        ax.set_ylabel(r'Torsion $\tau(t)$')
        ax_phi.set_ylabel(r'Control $\phi(t) = \int_{t_0}^t \tau(s)ds$')
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax_phi.yaxis.label.set_color("black")
        ax.tick_params(axis='x', colors="black")
        ax.tick_params(axis='y', colors="black")
        ax_phi.tick_params(axis='y', colors="black")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.legend([tau_line, phi_line], [tau_line.get_label(), phi_line.get_label()], loc="best")
        fig.tight_layout()
        ax.set_title(r"Torsion $\tau(t)$ and Control $\phi(t)$ over time")
        plt.show()

    def plot_position(self, samples = 4096, elev = 26, azim = 38):
        t_samples = jnp.linspace(self.t0, self.tf, samples)
        r_values = jax.vmap(self.position)(t_samples)
        time_cmap = LinearSegmentedColormap.from_list("curve_time", ["#999999", "#C73333"])

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection='3d')

        ax.plot(
            r_values[:, 0],
            r_values[:, 1],
            r_values[:, 2],
            color="0.75",
            linewidth=1.4,
            alpha=0.9,
            zorder=1,
        )
        scatter = ax.scatter(
            r_values[:, 0],
            r_values[:, 1],
            r_values[:, 2],
            c=t_samples,
            cmap=time_cmap,
            s=12,
            linewidths=0.0,
            zorder=2,
        )
        ax.scatter(*r_values[0], color="#999999", s=60, marker="o", edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(*r_values[-1], color="#C73333", s=80, marker="X", edgecolors="white", linewidths=0.8, zorder=3)

        mins = jnp.min(r_values, axis=0)
        maxs = jnp.max(r_values, axis=0)
        span = jnp.maximum(maxs - mins, 1e-6)
        ax.set_box_aspect(tuple(float(v) for v in (span / jnp.max(span))))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(r'Curve Position $r(t)$')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.82, label=r'$t$')
        fig.tight_layout()
        plt.show()

    def saveto(self, path: str):
        eq.tree_serialise_leaves(path, self)

    @classmethod
    def loadfrom(cls, path: str, *init_args, **init_kwargs) -> Self:
        return eq.tree_deserialise_leaves(path, cls(*init_args, **init_kwargs))

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
    ) -> Self:
        if optimizer is None: optimizer = optax.adam(lr)

        assert steps % chunk_size == 0, "steps must be divisible by chunk_size"

        if filter_spec is None: filter_spec = self.trainable_filter_spec()

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

            params, opt_state, key, losses = train_chunk(params, opt_state, key)

            if callback is not None:
                model = eq.combine(params, static)
                for loss in losses:
                    callback(step_counter, float(loss), model)
                    step_counter += 1

        return eq.combine(params, static)

    def optimize_lbfgs(
        self,
        cost_fn: Callable[["LearnableCurve", jnp.ndarray], jnp.ndarray],
        key: jnp.ndarray,
        steps: int = 200,
        chunk_size: int = 10,
        memory_size: int = 10,
        lr = None,
        scale_init_precond: bool = True,
        linesearch = None,
        filter_spec = None,
        callback=None,
    ) -> Self:
        assert steps % chunk_size == 0, "steps must be divisible by chunk_size"

        if filter_spec is None: filter_spec = self.trainable_filter_spec()

        params, static = eq.partition(self, filter_spec)

        if linesearch is None: optimizer = optax.lbfgs(learning_rate=lr, memory_size=memory_size, scale_init_precond=scale_init_precond,)
        else: optimizer = optax.lbfgs(learning_rate=lr, memory_size=memory_size, scale_init_precond=scale_init_precond, linesearch=linesearch,)

        opt_state = optimizer.init(params)
        value_dtype = optax.tree.get(opt_state, "value").dtype

        def loss_fn(params, key):
            value = cost_fn(eq.combine(params, static), key)
            return jnp.asarray(value, dtype=value_dtype)

        value_and_grad = optax.value_and_grad_from_state(loss_fn)

        @eq.filter_jit
        def train_chunk(params, opt_state, key):
            def body_fn(carry, _):
                params, opt_state, key = carry
                value, grad = value_and_grad(params, key, state=opt_state)
                updates, opt_state = optimizer.update(
                    grad,
                    opt_state,
                    params,
                    value=value,
                    grad=grad,
                    value_fn=loss_fn,
                    key=key,
                )
                params = optax.apply_updates(params, updates)
                return (params, opt_state, key), value

            (params, opt_state, key), losses = jax.lax.scan(
                body_fn,
                (params, opt_state, key),
                None,
                length=chunk_size,
            )

            return params, opt_state, key, losses

        step_counter = 0

        for _ in range(steps // chunk_size):
            params, opt_state, key, losses = train_chunk(params, opt_state, key)

            if callback is not None:
                model = eq.combine(params, static)
                for loss in losses:
                    callback(step_counter, float(loss), model)
                    step_counter += 1

        return eq.combine(params, static)


class InvertedCurve(LearnableCurve):
    """
    Note: Feasible invertibility requires that base.tangent(base.tf) == [0, 0, -1].
    Otherwise simulation will not hold up.
    """

    base: LearnableCurve

    def __init__(self, base: LearnableCurve):
        super().__init__(base.t0, base.tf)
        self.base = base

    def inverter(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.t0 + self.tf - t 

    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.base.position( self.inverter(t) )
    
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        return -self.base.velocity(self.inverter(t))
    
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.base.acceleration(self.inverter(t))
    
    def jerk(self, t: jnp.ndarray) -> jnp.ndarray:
        return -self.base.jerk(self.inverter(t))
