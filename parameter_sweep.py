from sys import argv
import numpy as np
import jax.numpy as jnp
import jax
from equinox import tree_deserialise_leaves
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

    errors = np.empty((runs, 4))

    iters = trange(runs) if use_callback else range(runs)

    key = jax.random.key(seed)
    curve = BezierCurve(32, [0, 0, 0], key)
    system = CompiledControls(curve)

    tree_deserialise_leaves('bezier.eqx', curve)

    for sweep_idx in trange(runs):

        paulis = jax.random.uniform(key, (3,), maxval=0.33)
        errors[sweep_idx, 0] = paulis[0].item()
        errors[sweep_idx, 1] = paulis[1].item()
        errors[sweep_idx, 2] = paulis[2].item()

        H_in = make_traceless_H(*paulis)

        diffrax_res = system(H_in)
        realized_unitary = jnp.asarray( diffrax_res.ys )[0]

        pow2 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 2)
        pow4 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 4)
        pow6 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 6)

        target_polynomial = pow2 / 6 + pow4 / 120 - pow6 / 5040
        expected = jax.scipy.linalg.expm(-1j * jnp.kron(target_polynomial, sig_z()))

        final_error : jnp.ndarray = jnp.linalg.matrix_norm( realized_unitary - expected, ord=2 )

        errors[sweep_idx, 3] = final_error.item()

    np.save('errors.npz', errors)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    p = ax.scatter( errors[:, 0], errors[:, 1], errors[:, 2], c=errors[:, 3], cmap='viridis' )
    cbar = fig.colorbar(p, ax=ax, pad=0.1)
    cbar.set_label(r'Error $ \left\| \exp\left(-i \left(\sum_{k=1}^d = c_k H^k \right) \otimes \sigma_z \right) - U_{actual}(T_g) \right\| $')

    ax.set_xlabel(r'$\sigma_x$')
    ax.set_ylabel(r'$\sigma_y$')
    ax.set_zlabel(r'$\sigma_z$')

    fig.savefig('error_sweep.png')

