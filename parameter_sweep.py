from sys import argv
import numpy as np
import jax.numpy as jnp
import jax
from equinox import tree_deserialise_leaves
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from time import sleep

from jaxcurve import CompiledControls, LearnableCurve, make_traceless_H, sig_z, sig_x, sig_y
from jaxcurveimpls import BezierCurve

if __name__ == "__main__":
    assert len(argv) == 4
    seed = int(argv[1])
    runs = int(argv[2])
    use_callback = bool(argv[3])

    errors = np.empty((runs, 4))

    iters = trange(runs) if use_callback else range(runs)

    key = jax.random.key(seed)
    curve : BezierCurve = tree_deserialise_leaves('bezier.eqx', BezierCurve(32, [0, 0, 0], key))

    key, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    coefs = jnp.stack([
        curve.magnus(2, k2, 8192),
        curve.magnus(3, k3, 8192),
        curve.magnus(4, k4, 8192),
        curve.magnus(5, k5, 8192),
        curve.magnus(6, k6, 8192)
    ])

    system = CompiledControls(curve)

    for sweep_idx in trange(runs):
        key, subkey = jax.random.split(key, 2)

        paulis = jax.random.uniform(subkey, (3,), maxval=0.3333)

        errors[sweep_idx, 0] = paulis[0].item()
        errors[sweep_idx, 1] = paulis[1].item()
        errors[sweep_idx, 2] = paulis[2].item()

        H_in = make_traceless_H(*paulis)

        diffrax_res = system(H_in)
        realized_unitary = jnp.asarray( diffrax_res.ys )[0]

        def step(carry, _):
            carry = carry @ H_in
            return carry, carry
        
        _, powers = jax.lax.scan(step, H_in, None, length=5)

        p = jnp.einsum('kc,kij->cij', coefs, powers)

        f_H = jnp.kron( p[0], sig_x() ) + jnp.kron( p[1], sig_y() ) + jnp.kron( p[2], sig_z() )

        expected = jax.scipy.linalg.expm(-1j * f_H)

        final_error : jnp.ndarray = jnp.linalg.matrix_norm( realized_unitary - expected, ord=2 )

        errors[sweep_idx, 3] = final_error.item()

    np.save('true_error', errors)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    p = ax.scatter( errors[:, 0], errors[:, 1], errors[:, 2], c=errors[:, 3], cmap='viridis' )

    ax.set_xlabel(r'$\sigma_x$')
    ax.set_ylabel(r'$\sigma_y$')
    ax.set_zlabel(r'$\sigma_z$')

    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Error')

    fig.savefig('error_sweep.png')
