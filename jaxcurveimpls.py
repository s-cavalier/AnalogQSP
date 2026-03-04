from jaxcurve import LearnableCurve, ArclenParameterize
import jax.numpy as jnp
import jax
import equinox as eq

class FourierCurve(LearnableCurve):
    Z: jnp.ndarray
    p_mat: jnp.ndarray = eq.field(static=True)
    null_M: jnp.ndarray = eq.field(static=True)
    frequencies: jnp.ndarray = eq.field(static=True)

    def __init__(self, frequencies: jax.Array, *fixed_pts: tuple[jax.Array, float, int] ):
        super().__init__(0, 1)

        frequencies = jnp.asarray(frequencies)
        self.frequencies = frequencies

        assert self.frequencies.shape[0] == len(fixed_pts)

        M = jnp.stack([
            jnp.concatenate([
                (frequencies ** d) * jnp.sin( frequencies * t + jnp.pi * d / 2 ),
                (frequencies ** d) * jnp.cos( frequencies * t + jnp.pi * d / 2)
            ])
            for _, t, d in fixed_pts
        ])

        R = jnp.asarray([ r for r, _, _ in fixed_pts ])

        self.p_mat = jnp.linalg.lstsq(M, R, rcond=None)[0]

        Q, _ = jnp.linalg.qr(M.T, mode="complete")
        self.null_M = Q[:, M.shape[0]:]

        self.Z = jnp.zeros((self.null_M.shape[1], 3))

    def get_X(self):
        return self.p_mat + self.null_M @ self.Z

    def position(self, t: float):
        N = self.frequencies.shape[0]
        X = self.get_X()

        time_freq = self.frequencies * t
        return jnp.sum(X[:N] * jnp.sin(time_freq)[:, None] + X[N:2*N] * jnp.cos(time_freq)[:, None], axis=0)

    def velocity(self, t: float):
        N = self.frequencies.shape[0]
        X = self.get_X()

        time_freq = self.frequencies * t
        return jnp.sum(self.frequencies[:, None] * ( X[:N] * jnp.cos(time_freq)[:, None] - X[N:2*N] * jnp.sin(time_freq)[:, None] ), axis=0)
        
    def acceleration(self, t: float):
        N = self.frequencies.shape[0]
        X = self.get_X()

        time_freq = self.frequencies * t
        return jnp.sum((self.frequencies ** 2)[:, None] * ( -X[:N] * jnp.sin(time_freq)[:, None] - X[N:2*N] * jnp.cos(time_freq)[:, None] ), axis=0)
