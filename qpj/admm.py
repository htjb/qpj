"""ADMM solver for QP with derivative sign constraints."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


@jax.jit
def admm_qp(
    Q, q, G, tol=1e-4, max_iters=50000
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Full ADMM solver for QP.

    Args:
        Q: Quadratic term (N,N)
        q: Linear term (N,)
        G: Constraint matrix (K,n,N)
        tol: convergence tolerance
        max_iters: maximum iterations

    Returns:
        p: final polynomial coefficients
        s: final slack
        u: final dual
        n_iter: number of iterations run
    """
    K, n, N = G.shape
    g_norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)
    G = G / g_norm
    G2d = G.reshape(K * n, N)

    L, lower = cho_factor(Q)
    p_init = cho_solve((L, lower), -q)

    # Initialize state tuple: (p, s, u, iteration, converged)
    p0 = p_init
    s0 = jnp.maximum(-G2d @ p_init, 0.0)
    u0 = jnp.zeros(G2d.shape[0])
    n_iter0 = 0
    converged0 = False

    Q_norm = jnp.linalg.norm(q, ord=2)  # scale of linear term
    s_norm = jnp.linalg.norm(
        G @ p_init, ord=2
    )  # scale of constraint violation
    s_norm = jnp.maximum(s_norm, 1.0)  # avoid division by tiny numbers
    rho = 0.1*Q_norm / s_norm

    state = (p0, s0, u0, rho, n_iter0, converged0)

    # Condition function
    def cond_fn(state):
        _, _, _, _, n_iter, converged = state
        return jnp.logical_and(n_iter < max_iters, ~converged)

    # Body function
    def body_fn(state):
        p, s, u, rho, n_iter, _ = state

        A = Q + rho * (G2d.T @ G2d)
        L, lower = cho_factor(A)

        # --- Primal update ---
        rhs = -q - rho * G2d.T @ (s + u)

        pnew = cho_solve((L, lower), rhs)

        # --- Slack update ---
        snew = jnp.maximum(-(G2d @ pnew + u), 0.0)

        # --- Dual update ---
        unew = u + G2d @ pnew + snew

        # --- Check convergence ---
        primal_resid = jnp.linalg.norm(G2d @ pnew + snew)
        dual_resid = jnp.linalg.norm(-rho * G2d.T @ (snew - s))

        k = 5
        update_rho = jnp.logical_and(n_iter % k == 0, n_iter > 0)

        rhonew = jax.lax.cond(
            update_rho,
            lambda rho: jax.lax.cond(
                primal_resid > 10 * dual_resid,
                lambda rho: rho * 2,
                lambda rho: jax.lax.cond(
                    dual_resid > 10 * primal_resid,
                    lambda rho: rho / 2,
                    lambda rho: rho,
                    operand=rho,
                ),
                operand=rho,
            ),
            lambda rho: rho,
            operand=rho,
        )

        delta_p = jnp.linalg.norm(pnew - p)
        converged = jnp.logical_and(primal_resid < tol, delta_p < tol)

        return (pnew, snew, unew, rhonew, n_iter + 1, converged)

    # Run the while loop
    p_final, s_final, u_final, rho_final, n_iter_final, converged = (
        jax.lax.while_loop(cond_fn, body_fn, state)
    )
    viol = jnp.max(jnp.abs(G2d @ p_final + s_final))
    return p_final, n_iter_final, converged, s_final, u_final, viol
