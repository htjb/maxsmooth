import jax

# function to generate derivatives
def make_derivative_functions(f, max_order):
    derivs = [f]  
    for n in range(1, max_order):
        derivs.append(jax.grad(derivs[-1], argnums=0))
    return derivs

def derivative_prefactors(f, x, norm_x, norm_y, params, max_order):
    """Return list of derivative matrices G[m] mapping params -> m-th derivative at all x."""
    
    Gs = []
    for m in range(max_order):
        df_dx = f
        # take m-th derivative w.r.t x
        for _ in range(m):
            prev_df_dx = df_dx
            def df_dx(xi, norm_x, norm_y, p, prev_df_dx=prev_df_dx):
                return jax.grad(prev_df_dx, argnums=0)(xi, norm_x, norm_y, p)
        # Jacobian w.r.t params
        Gm = jax.vmap(lambda xi: jax.jacobian(df_dx, argnums=3)(xi, norm_x, norm_y, params))(x)
        Gs.append(Gm)
    return Gs