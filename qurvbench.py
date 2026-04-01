from jaxcurve import ArclenParameterize
from jaxcurveimpls import BezierCurve
import jax
import jax.numpy as jnp

key = jax.random.key(4443)

curve = BezierCurve(16, [4, 0, 0], key)

print( curve.position(1) - curve.position(0) )
print( jnp.linalg.norm(curve.magnus(1, key, 8192)) )
