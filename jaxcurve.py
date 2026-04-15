import equinox as eq
import jax
import jax.numpy as jnp
import optax
import diffrax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from warnings import warn
from abc import abstractmethod
from typing import Callable, Self
from functools import lru_cache

def _cumulative_trapezoid(values: jnp.ndarray, dx):
    partial = jnp.cumsum((values[1:] + values[:-1]) * (0.5 * dx), axis=0)
    return jnp.concatenate([jnp.zeros_like(values[:1]), partial], axis=0)


@lru_cache(maxsize=None)
def _gauss_legendre_01(order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1.0) * 0.5, weights * 0.5

"""
JAX-based workflow is slightly different.
Implement a curve that inherits from LearnableCurve (so, implement position, velocity, acceleration, and jerk).
You MUST implement position at the minimum. Velocity, acceleration, and jerk use equinox/jax autodifferentiation as fallback if the implementations
are too complex or too trivial. A warning will be made at __init__.
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
        if cls.velocity is LearnableCurve.velocity: warn(f"{cls.__name__}.velocity not implemented, using equinox.filter_grad fallback")
        if cls.acceleration is LearnableCurve.acceleration: warn(f"{cls.__name__}.acceleration not implemented, using equinox.filter_grad fallback")
        if cls.jerk is LearnableCurve.jerk: warn(f"{cls.__name__}.jerk not implemented, using equinox.filter_grad fallback")

    @abstractmethod
    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        pass

    """
    Should be overloaded, but if not possible or non-trivial, equinox.filter_grad is used in place
    """
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        return eq.filter_grad(self.position)(t)

    """
    Should be overloaded, but if not possible or non-trivial, equinox.filter_grad is used in place
    """
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        return eq.filter_grad(self.velocity)(t)

    """
    Should be overloaded, but if not possible or non-trivial, equinox.filter_grad is used in place
    """
    def jerk(self, t: jnp.ndarray) -> jnp.ndarray:
        return eq.filter_grad(self.acceleration)(t)

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

        d3r_x, d3r_y, _ = self.jerk(0)
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
    def to_time(self, s, resolution = 4096, newton_steps = 8):
        s = jnp.asarray(s)
        flat_s = s.reshape(-1)
        x = jnp.linspace(0.0, 1.0, resolution)
        dx = 1.0 / (resolution - 1)

        def speed(t): return jnp.linalg.norm(self.velocity(t), axis=-1)

        def arclength_to_scalar(t):
            t = jnp.clip(jnp.asarray(t), self.t0, self.tf)
            tau = self.t0 + (t - self.t0) * x
            speed_values = jax.vmap(speed)(tau)
            return (t - self.t0) * jnp.sum((speed_values[:-1] + speed_values[1:]) * (0.5 * dx))

        total_length = arclength_to_scalar(self.tf)

        s_target = jnp.clip(flat_s, 0.0, total_length)
        t_init = self.t0 + (self.tf - self.t0) * s_target / (total_length + 1e-8)

        def body(_, t):
            s_current = jax.vmap(arclength_to_scalar)(t)
            speed_current = jax.vmap(speed)(t)
            t = t - (s_current - s_target) / (speed_current + 1e-8)
            return jnp.clip(t, self.t0, self.tf)

        t = jax.lax.fori_loop(0, newton_steps, body, t_init)
        return t.reshape(s.shape)

    @eq.filter_jit
    def arclen_linspace(self, samples = 4096, resolution = 4096, newton_steps = 8):
        """Computes linspace(0, arclen, samples) and maps each to the corresponding time value for evenly spaced arclen points as time points"""

        return self.to_time( jnp.linspace(0, self.arclength(samples), samples), resolution, newton_steps )

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
    def magnus_mc(self, order, key, samples = 4096):
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

    @eq.filter_jit
    def magnus(self, n: int, samples = 4096):
        if n < 1:
            raise ValueError("n must be at least 1")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        if n == 1:
            return self.position(self.tf) - self.position(self.t0)

        time = jnp.linspace(self.t0, self.tf, samples)
        T = jax.vmap(self.velocity)(time)
        dt = (self.tf - self.t0) / (samples - 1)

        complex_dtype = jnp.result_type(T.dtype, jnp.complex64)
        T = T.astype(complex_dtype)

        x_order = (n + 1) // 2
        x_nodes_np, x_weights_np = _gauss_legendre_01(x_order)
        x_nodes = jnp.asarray(x_nodes_np, dtype=T.real.dtype)
        x_weights = jnp.asarray(x_weights_np, dtype=T.real.dtype)
        shift = x_nodes - 1.0

        n_x = x_nodes.shape[0]
        Y_a = jnp.zeros((samples, n_x), dtype=complex_dtype)
        Y_b = jnp.broadcast_to(T[:, None, :], (samples, n_x, 3))

        for _ in range(1, n):
            prefix_a = _cumulative_trapezoid(Y_a, dt)
            prefix_b = _cumulative_trapezoid(Y_b, dt)

            total_a = prefix_a[-1]
            total_b = prefix_b[-1]

            S_a = prefix_a + shift[None, :] * total_a[None, :]
            S_b = prefix_b + shift[None, :, None] * total_b[None, :, :]

            Y_a = jnp.einsum("td,txd->tx", T, S_b)
            Y_b = S_a[:, :, None] * T[:, None, :] + 1j * jnp.cross(T[:, None, :], S_b, axis=-1)

        final_b = _cumulative_trapezoid(Y_b, dt)[-1]
        value = jnp.sum(x_weights[:, None] * final_b, axis=0)

        return jnp.real((1.0j ** (1 - n)) * value)

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

    def plot_torsion(self, samples = 4096):
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
    ):
        assert steps % chunk_size == 0, "steps must be divisible by chunk_size"

        if filter_spec is None:
            filter_spec = self.trainable_filter_spec()

        params, static = eq.partition(self, filter_spec)

        if linesearch is None:
            optimizer = optax.lbfgs(
                learning_rate=lr,
                memory_size=memory_size,
                scale_init_precond=scale_init_precond,
            )
        else:
            optimizer = optax.lbfgs(
                learning_rate=lr,
                memory_size=memory_size,
                scale_init_precond=scale_init_precond,
                linesearch=linesearch,
            )

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

def _default_complex_dtype():
    return jnp.asarray(1j).dtype

@jax.jit
def sig_x():
    return jnp.array([[0, 1], [1, 0]], dtype=_default_complex_dtype())

@jax.jit
def sig_y():
    return jnp.array([[0, -1j], [1j, 0]], dtype=_default_complex_dtype())

@jax.jit
def sig_z():
    return jnp.array([[1, 0], [0, -1]], dtype=_default_complex_dtype())

@jax.jit
def paulis():
    return jnp.stack([sig_x(), sig_y(), sig_z()])

@jax.jit
def make_traceless_H(a, b, c):
    complex_dtype = jnp.result_type(a, b, c, _default_complex_dtype())
    sx = sig_x().astype(complex_dtype)
    sy = sig_y().astype(complex_dtype)
    sz = sig_z().astype(complex_dtype)
    return sx * a + sy * b + sz * c

class CompiledControls( eq.Module ):
    curve: LearnableCurve
    U_I: diffrax.Solution
    samples: int = eq.field(static=True)
    complex_dtype: jnp.dtype = eq.field(static=True)

    def __init__(self, curve: LearnableCurve, samples=4096):
        self.curve = curve
        self.samples = samples
        curve_dtype = jnp.asarray(self.curve.position(self.curve.t0)).dtype
        self.complex_dtype = jnp.result_type(curve_dtype, _default_complex_dtype())

        @eq.filter_jit
        def drive_system(t: float, U: jnp.ndarray, args):
            curve : LearnableCurve = args['curve']
            complex_dtype = args["complex_dtype"]

            speed = jnp.linalg.norm( curve.velocity(t) )
            curvature = curve.curvature(t).astype(complex_dtype)
            phi = curve.phi(t).astype(complex_dtype)

            Omega = 0.5 * jnp.asarray(speed, dtype=complex_dtype) * curvature

            sx = sig_x().astype(complex_dtype)
            sy = sig_y().astype(complex_dtype)
            H = Omega * ( jnp.cos(phi) * sx + jnp.sin(phi) * sy )

            return -1j * H @ U

        self.U_I = diffrax.diffeqsolve( 
            diffrax.ODETerm(drive_system), 
            diffrax.Dopri5(), 
            self.curve.t0, self.curve.tf,
            dt0=(self.curve.tf - self.curve.t0)/self.samples,
            y0=jnp.eye(2, dtype=self.complex_dtype),
            args={ "curve" : self.curve, "complex_dtype": self.complex_dtype },
            saveat=diffrax.SaveAt(dense=True)
        )

    def __call__(self, H_in : jnp.ndarray):

        @eq.filter_jit
        def interaction_system(t: float, U: jnp.ndarray, args):
            U_I : diffrax.Solution = args["U_I"]
            curve : LearnableCurve = args["curve"]
            complex_dtype = args["complex_dtype"]

            ui_t : jnp.ndarray = jnp.asarray(U_I.evaluate(t))

            H_input : jnp.ndarray = jnp.asarray(args["H_input"], dtype=complex_dtype)

            speed = jnp.asarray(jnp.linalg.norm( curve.velocity(t) ), dtype=complex_dtype)
            sz = sig_z().astype(complex_dtype)

            ham = speed * jnp.kron( H_input, ui_t.T.conj() @ sz @ ui_t )

            return -1j * ham @ U
        
        H_in = jnp.asarray(H_in)
        y0_dtype = jnp.result_type(H_in.dtype, self.complex_dtype)
        result = diffrax.diffeqsolve(
            diffrax.ODETerm(interaction_system),
            diffrax.Dopri5(),
            self.curve.t0, self.curve.tf, 
            dt0=(self.curve.tf - self.curve.t0) / self.samples,
            y0=jnp.eye( H_in.shape[0] * 2, dtype=y0_dtype ),
            args={
                "U_I" : self.U_I,
                "H_input" : H_in,
                "curve" : self.curve,
                "complex_dtype": y0_dtype,
            },
            #progress_meter=diffrax.TqdmProgressMeter()
        )

        return result



class ArclenParameterize(LearnableCurve):
    """
    Depracated
    Can be used, but increases runtime a good bit and doesn't really provide meaningful advantage
    Instead we just use arclen parameterized calculations

    Important to note - calculation of magnus expansion for AnalogQSP is invariant under arc-len parameterization, so we can just not do it.
    A sketch proof goes something like considering the nested integral formed by the 1969 Annals of Physics paper. 
    Each integrand contributes one T(t_k) to the ordered geometric product of T(t_n)T(t_{n-1}) ... T(t_1)
    We know then T(t_k) = r'(t_k) / || r'(t_k) ||, so we get the ordered geometric product r'(t_n)...r'(t_1)/( || r'(t_n) || ... || r'(t_1) || )
    Since each term is defined w.r.t. dt_k, we can do a change of variables toget dx_k = || r'(t_k) || dt_k
    So we end up with

    r'(x_n)...r'(x_1) / ( || r'(x_n) || ... || r'(x_1) || ) || r'(x_n) || ... || r'(x_1) || dx_n ... dx_1 
    
    which is just r'(x_n)...r'(x_1). So arclength parameterization doesn't really buy much as in the case of error correction.
    """
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
    
