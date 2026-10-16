import jax
import jax.numpy as jnp
import equinox as eq
import diffrax

from curve import LearnableCurve

def _default_complex_dtype(): return jnp.asarray(1j).dtype

@jax.jit
def sig_x(): return jnp.array([[0, 1], [1, 0]], dtype=_default_complex_dtype())

@jax.jit
def sig_y(): return jnp.array([[0, -1j], [1j, 0]], dtype=_default_complex_dtype())

@jax.jit
def sig_z(): return jnp.array([[1, 0], [0, -1]], dtype=_default_complex_dtype())

@jax.jit
def paulis(): return jnp.stack([sig_x(), sig_y(), sig_z()])

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