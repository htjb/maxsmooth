import jax


# function to generate derivatives
def make_derivative_functions(f, max_order):
    derivs = [f]  
    for n in range(1, max_order):
        derivs.append(jax.grad(derivs[-1], argnums=0))
    return derivs