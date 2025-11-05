import jax

# function to generate derivatives
def make_derivative_functions(f, max_order):
    derivs = [f]  
    for n in range(1, max_order):
        derivs.append(jax.grad(derivs[-1], argnums=0))
    return derivs

def derivative_prefactors(f, x, norm_x, norm_y, params, max_order):
    Gs = []
    df_dx = f  # start from f

    for m in range(max_order):
        # Jacobian of current derivative w.r.t parameters
        Gm = jax.vmap(lambda xi: jax.jacobian(df_dx, argnums=3)(xi, norm_x, norm_y, params))(x)
        Gs.append(Gm)

        # Prepare next derivative wrt x
        df_dx = jax.grad(df_dx, argnums=0)
    return Gs