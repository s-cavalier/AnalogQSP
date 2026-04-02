from sys import argv
import numpy as np
import jax.numpy as jnp
import jax
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from time import sleep

from jaxcurve import CompiledControls, LearnableCurve, make_traceless_H, sig_z
from jaxcurveimpls import BezierCurve

if __name__ == "__main__":
    assert len(argv) == 4
    seed = int(argv[1])
    runs = int(argv[2])
    use_callback = bool(argv[3])

    discard = jnp.empty((1,)) # discard so the jax log comes before this
    sleep(1) 

    errors = np.empty((runs, 4))

    rng = np.random.default_rng(seed)

    iters = trange(runs) if use_callback else range(runs)

    for sweep_idx in trange(runs):
        key = jax.random.key( rng.integers(0, 2_000_000_000).item() )
        curve = BezierCurve(32, [0, 0, 0], key)


        def cost_fn( c: LearnableCurve, k: jnp.ndarray ):
            m2 = 2048 * jnp.linalg.norm(c.magnus(2, k, 10_000) - jnp.array([0, 0, -1/6])) # -1/6
            m3 = 8192 * jnp.linalg.norm(c.magnus(3, k, 10_000) - jnp.array([0, 0, 0])) # 0
            m4 = 4096 * jnp.linalg.norm(c.magnus(4, k, 10_000) - jnp.array([0, 0, 1/120])) # 1/120
            m5 = 8192 * jnp.linalg.norm(c.magnus(5, k, 10_000) - jnp.array([0, 0, 0])) # 0
            m6 = 16384 * jnp.linalg.norm(c.magnus(6, k, 10_000) - jnp.array([0, 0, -1/5040])) # 1/120

            return m2 + m3 + m4 + m5 + m6

        STEPS = 15_000
        
        bar = tqdm(total=STEPS, leave=False)

        def callback( step, loss, curve ): bar.update(1)

        used_callback = callback if use_callback else None

        curve : BezierCurve = curve.optimize(cost_fn, key, steps=STEPS, lr=1e-3, callback=used_callback)

        paulis = jax.random.uniform(key, (3,), maxval=0.33)
        errors[sweep_idx, 0] = paulis[0].item()
        errors[sweep_idx, 1] = paulis[1].item()
        errors[sweep_idx, 2] = paulis[2].item()

        H_in = make_traceless_H(*paulis)

        system = CompiledControls(curve)
        diffrax_res = system(H_in)

        realized_unitary = jnp.asarray( diffrax_res.ys )[0]

        pow2 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 2)
        pow4 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 4)
        pow6 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 6)

        expected = jax.scipy.linalg.expm(-1j * jnp.kron(
            jnp.eye(2) - 1j * pow2/6 - 1j * pow4/120 - 1j * pow6/5040,
            sig_z()
        ))

        final_error : jnp.ndarray = jnp.linalg.norm( realized_unitary - expected )

        errors[sweep_idx, 3] = final_error.item()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    p = ax.scatter( errors[:, 0], errors[:, 1], errors[:, 2], c=errors[:, 3], cmap='viridis' )
    cbar = fig.colorbar(p, ax=ax, pad=0.1)
    cbar.set_label(r'Error $ \left\| \exp\left(-i \left(\sum_{k=1}^d = c_k H^k \right) \otimes \sigma_z \right) - U_{actual}(T_g) \right\| $')

    ax.set_xlabel(r'$\sigma_x$')
    ax.set_ylabel(r'$\sigma_y$')
    ax.set_zlabel(r'$\sigma_z$')

    fig.savefig('error_sweep.png')


