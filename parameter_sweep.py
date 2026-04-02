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

    errors = np.empty((runs, 3))

    iters = trange(runs) if use_callback else range(runs)

    key = jax.random.key(seed)
    curve = BezierCurve(32, [0, 0, 0], key)
    system = CompiledControls(curve)

    tree_deserialise_leaves('bezier.eqx', curve)

    for sweep_idx in trange(runs):
        key, subkey = jax.random.split(key, 2)

        paulis = jax.random.uniform(subkey, (2,), maxval=0.4999999)
        
        errors[sweep_idx, 0] = paulis[0].item()
        errors[sweep_idx, 1] = paulis[1].item()

        H_in = make_traceless_H(*paulis, 0)

        diffrax_res = system(H_in)
        realized_unitary = jnp.asarray( diffrax_res.ys )[0]

        pow2 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 2)
        pow4 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 4)
        pow6 : jnp.ndarray = jnp.linalg.matrix_power(H_in, 6)

        target_polynomial = pow2 / 6 + pow4 / 120 - pow6 / 5040
        expected = jax.scipy.linalg.expm(-1j * jnp.kron(target_polynomial, sig_z()))

        final_error : jnp.ndarray = jnp.linalg.matrix_norm( realized_unitary - expected, ord=2 )

        errors[sweep_idx, 2] = final_error.item()

    np.save('errors.npz', errors)

    data = np.load

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    p = ax.scatter( errors[:, 0], errors[:, 1], errors[:, 2], c=errors[:, 3])

    ax.set_xlabel(r'$\sigma_x$')
    ax.set_ylabel(r'$\sigma_y$')
    ax.set_zlabel(r'$\sigma_z$')

    fig.savefig('error_sweep2d.png')

