import math
import numpy as np

import equinox as eq
import jax
import jax.numpy as jnp

from jaxcurve import LearnableCurve


class BezierCurve(LearnableCurve):
    num_control_points: int = eq.field(static=True)
    endpoint: jax.Array
    free_control_points: jax.Array

    # below are non trainable since they are integer arrays
    binomial_coeffs: jax.Array
    velocity_binomial_coeffs: jax.Array
    acceleration_binomial_coeffs: jax.Array
    jerk_binomial_coeffs: jax.Array

    def __init__(
        self,
        num_control_points: int,
        endpoint: jax.Array,
        key: jax.Array | None = None,
        seed: int | None = None,
    ):
        super().__init__(0., 1.)

        if num_control_points <= 6: raise ValueError("num_control_points must be greater than 6")
        if key is not None and seed is not None: raise ValueError("pass at most one of key or seed")

        endpoint = jnp.asarray(endpoint, dtype=jnp.float64)
        if endpoint.shape != (3,): raise ValueError("endpoint must be a 3D point")

        degree = num_control_points - 1
        line = jnp.linspace(0.0, 1.0, num_control_points, dtype=endpoint.dtype)[:, None] * endpoint[None, :]
        free_indices = [1, *range(3, num_control_points - 3), num_control_points - 2]
        base_free_points = line[jnp.asarray(free_indices)]

        scale = max(float(jnp.linalg.norm(endpoint)), 1.0)
        if key is None:
            if seed is None:
                seed = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            key = jax.random.key(int(seed))

        noise = jax.random.normal(key, shape=base_free_points.shape, dtype=endpoint.dtype)
        randomized_free_points = base_free_points + 0.25 * scale * noise

        self.num_control_points = int(num_control_points)
        self.endpoint = endpoint
        self.free_control_points = randomized_free_points
        self.binomial_coeffs = jnp.asarray(
            [math.comb(degree, i) for i in range(degree + 1)],
            dtype=jnp.uint32,
        )
        self.velocity_binomial_coeffs = jnp.asarray(
            [math.comb(degree - 1, i) for i in range(degree)],
            dtype=jnp.uint32,
        )
        self.acceleration_binomial_coeffs = jnp.asarray(
            [math.comb(degree - 2, i) for i in range(degree - 1)],
            dtype=jnp.uint32,
        )
        self.jerk_binomial_coeffs = jnp.asarray(
            [math.comb(degree - 3, i) for i in range(degree - 2)],
            dtype=jnp.uint32,
        )

    def trainable_filter_spec(self):
        mask = super().trainable_filter_spec()
        return eq.tree_at(lambda m: m.endpoint, mask, replace=False)

    def control_points(self):
        p0 = jnp.zeros((1, 3), dtype=self.endpoint.dtype)
        p1 = self.free_control_points[:1]
        p2 = 2.0 * p1 - p0
        middle = self.free_control_points[1:-1]
        pn1 = self.free_control_points[-1:]
        pn = self.endpoint[None, :]
        pn2 = 2.0 * pn1 - pn

        return jnp.concatenate([p0, p1, p2, middle, pn2, pn1, pn], axis=0)

    def _bernstein_basis(self, t, coeffs):
        t = jnp.asarray(t, dtype=self.endpoint.dtype)
        degree = coeffs.shape[0] - 1
        powers = jnp.arange(degree + 1)
        return coeffs * (t[..., None] ** powers) * ((1.0 - t)[..., None] ** (degree - powers))

    def position(self, t):
        basis = self._bernstein_basis(t, self.binomial_coeffs)
        return jnp.einsum("...i,ij->...j", basis, self.control_points())

    def velocity(self, t):
        degree = self.num_control_points - 1
        control_diffs = self.control_points()[1:] - self.control_points()[:-1]
        basis = self._bernstein_basis(t, self.velocity_binomial_coeffs)
        return degree * jnp.einsum("...i,ij->...j", basis, control_diffs)

    def acceleration(self, t):
        degree = self.num_control_points - 1
        control_points = self.control_points()
        second_diffs = control_points[2:] - 2.0 * control_points[1:-1] + control_points[:-2]
        basis = self._bernstein_basis(t, self.acceleration_binomial_coeffs)
        return degree * (degree - 1) * jnp.einsum("...i,ij->...j", basis, second_diffs)

    def jerk(self, t):
        degree = self.num_control_points - 1
        control_points = self.control_points()
        third_diffs = (
            control_points[3:]
            - 3.0 * control_points[2:-1]
            + 3.0 * control_points[1:-2]
            - control_points[:-3]
        )
        basis = self._bernstein_basis(t, self.jerk_binomial_coeffs)
        return degree * (degree - 1) * (degree - 2) * jnp.einsum("...i,ij->...j", basis, third_diffs)
