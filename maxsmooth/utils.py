from jax.scipy.linalg import cholesky
from jax import numpy as jnp

def is_positive_definite_cholesky(A: jnp.ndarray) -> bool:
  """
  Checks if a symmetric matrix A is positive definite using Cholesky factorization.

  parameters:
    -----------
    A : jnp.ndarray
      Symmetric matrix to be checked.
  """
  try:
    _ = cholesky(A, lower=True, check_finite=True)
    return True
  except ValueError:
    return False
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return False