import math

import equinox as eq
import jax
import jax.numpy as jnp

from jaxcurve import LearnableCurve


def _binomial_coefficients(degree: int, dtype) -> jnp.ndarray:
    return jnp.asarray([math.comb(degree, i) for i in range(degree + 1)], dtype=dtype)


def _evaluate_bezier(t: jnp.ndarray, control_points: jnp.ndarray, coefficients: jnp.ndarray) -> jnp.ndarray:
    degree = control_points.shape[0] - 1
    t = jnp.asarray(t, dtype=control_points.dtype)
    exponents = jnp.arange(degree + 1)
    basis = (
        coefficients
        * jnp.power(t[..., None], exponents)
        * jnp.power(1.0 - t[..., None], degree - exponents)
    )
    return jnp.einsum("...n,nc->...c", basis, control_points)


class BezierCurve(LearnableCurve):
    num_control_points: int = eq.field(static=True)
    endpoint: jnp.ndarray
    free_control_points: jnp.ndarray
    _position_coefficients: jnp.ndarray = eq.field(repr=False)
    _velocity_coefficients: jnp.ndarray = eq.field(repr=False)
    _acceleration_coefficients: jnp.ndarray = eq.field(repr=False)
    _jerk_coefficients: jnp.ndarray = eq.field(repr=False)

    def __init__(
        self,
        num_control_points: int,
        endpoint,
        key: jax.Array | None = None,
        init_scale: float = 0.1,
    ):
        if num_control_points < 6:
            raise ValueError("num_control_points must be at least 6 to satisfy the endpoint curvature constraints.")

        endpoint = jnp.asarray(endpoint)
        if endpoint.shape != (3,):
            raise ValueError("endpoint must have shape (3,).")

        self.num_control_points = int(num_control_points)
        self.endpoint = endpoint

        degree = self.num_control_points - 1
        dtype = endpoint.dtype
        initial_velocity = jnp.array([0.0, 0.0, 1.0], dtype=dtype)
        first = initial_velocity / jnp.asarray(degree, dtype=dtype)
        second = 2.0 * first

        middle_count = max(self.num_control_points - 6, 0)
        penultimate = endpoint - first
        interior_times = jnp.linspace(
            0.0,
            1.0,
            middle_count + 2,
            dtype=dtype,
        )[1:-1]
        middle = second[None, :] + interior_times[:, None] * (penultimate - second)[None, :]
        free_control_points = jnp.concatenate([middle, penultimate[None, :]], axis=0)

        if key is not None:
            noise_scale = jnp.asarray(init_scale, dtype=dtype) * jnp.maximum(
                jnp.linalg.norm(endpoint),
                jnp.asarray(1.0, dtype=dtype),
            )
            free_control_points = free_control_points + noise_scale * jax.random.normal(
                key,
                free_control_points.shape,
                dtype=dtype,
            )

        self.free_control_points = free_control_points
        self._position_coefficients = _binomial_coefficients(degree, dtype)
        self._velocity_coefficients = _binomial_coefficients(degree - 1, dtype)
        self._acceleration_coefficients = _binomial_coefficients(degree - 2, dtype)
        self._jerk_coefficients = _binomial_coefficients(degree - 3, dtype)

        super().__init__(0.0, 1.0)

    def _control_points(self) -> jnp.ndarray:
        degree = self.num_control_points - 1
        dtype = self.free_control_points.dtype
        first = (jnp.array([0.0, 0.0, 1.0], dtype=dtype) / jnp.asarray(degree, dtype=dtype))[None, :]
        second = 2.0 * first
        middle = self.free_control_points[:-1]
        penultimate = self.free_control_points[-1:]
        origin = jnp.zeros((1, 3), dtype=self.free_control_points.dtype)
        endpoint = self.endpoint[None, :]

        antepenultimate = 2.0 * penultimate - endpoint

        return jnp.concatenate(
            [origin, first, second, middle, antepenultimate, penultimate, endpoint],
            axis=0,
        )

    def trainable_filter_spec(self):
        spec = super().trainable_filter_spec()
        spec = eq.tree_at(lambda tree: tree.endpoint, spec, replace=False)
        spec = eq.tree_at(lambda tree: tree._position_coefficients, spec, replace=False)
        spec = eq.tree_at(lambda tree: tree._velocity_coefficients, spec, replace=False)
        spec = eq.tree_at(lambda tree: tree._acceleration_coefficients, spec, replace=False)
        spec = eq.tree_at(lambda tree: tree._jerk_coefficients, spec, replace=False)
        return spec

    @eq.filter_jit
    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        return _evaluate_bezier(t, self._control_points(), self._position_coefficients)

    @eq.filter_jit
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        control_points = self._control_points()
        degree = self.num_control_points - 1
        velocity_control_points = degree * (control_points[1:] - control_points[:-1])
        return _evaluate_bezier(t, velocity_control_points, self._velocity_coefficients)

    @eq.filter_jit
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        control_points = self._control_points()
        degree = self.num_control_points - 1
        acceleration_control_points = degree * (degree - 1) * (
            control_points[2:] - 2.0 * control_points[1:-1] + control_points[:-2]
        )
        return _evaluate_bezier(t, acceleration_control_points, self._acceleration_coefficients)

    @eq.filter_jit
    def jerk(self, t: jnp.ndarray) -> jnp.ndarray:
        control_points = self._control_points()
        degree = self.num_control_points - 1
        jerk_control_points = degree * (degree - 1) * (degree - 2) * (
            control_points[3:]
            - 3.0 * control_points[2:-1]
            + 3.0 * control_points[1:-2]
            - control_points[:-3]
        )
        return _evaluate_bezier(t, jerk_control_points, self._jerk_coefficients)
