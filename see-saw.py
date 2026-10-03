import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cvxpy as cp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ============================================================
# Noisy Pauli POVMs
# ============================================================

def povms_eta(eta):
    I = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    pauli = [sx, sy, sz]

    M = {}

    for x in range(3):
        M[x, 0] = (I + eta * pauli[x]) / 2
        M[x, 1] = (I - eta * pauli[x]) / 2

    return M


# ============================================================
# Assemblage
# ============================================================

def partial_trace_A(rho_AB, dA, dB):
    rho4 = rho_AB.reshape(dA, dB, dA, dB)
    rho_B = np.zeros((dB, dB), dtype=complex)

    for i in range(dB):
        for j in range(dB):
            for a in range(dA):
                rho_B[i, j] += rho4[a, i, a, j]

    return (rho_B + rho_B.conj().T) / 2


def assemblage_from_state(M, rho_AB):
    num_x = max(x for x, a in M.keys()) + 1
    num_a = max(a for x, a in M.keys()) + 1
    dA = M[0, 0].shape[0]
    dB = rho_AB.shape[0] // dA
    I_B = np.eye(dB, dtype=complex)

    assemblage = {}

    for x in range(num_x):
        for a in range(num_a):
            temp = np.kron(M[x, a], I_B) @ rho_AB
            assemblage[x, a] = partial_trace_A(temp, dA, dB)

    return assemblage


# ============================================================
# Star-incompatibility weight
# ============================================================

def solve_weight(M, x_star):
    x_values = {x for x, a in M.keys()}
    a_values = {a for x, a in M.keys()}
    num_x = len(x_values)
    num_a = len(a_values)

    d = M[0, 0].shape[0]
    I = np.eye(d, dtype=complex)

    s = cp.Variable()

    F = {}
    for x in range(num_x):
        for a in range(num_a):
            F[x, a] = cp.Variable((d, d), hermitian=True)

    G = {}
    for x in range(num_x):
        for a in range(num_a):
            for e in range(num_a):
                G[x, a, e] = cp.Variable((d, d), hermitian=True)

    constraints = []

    for x in range(num_x):
        for a in range(num_a):
            constraints.append(
                sum(G[x, a, e] for e in range(num_a)) == F[x, a]
            )

    for x in range(num_x):
        for e in range(num_a):
            constraints.append(
                sum(G[x, a, e] for a in range(num_a)) == F[x_star, e]
            )

    for x in range(num_x):
        constraints.append(
            sum(F[x, a] for a in range(num_a)) == s * I
        )

    for x in range(num_x):
        for a in range(num_a):
            constraints.append(M[x, a] - F[x, a] >> 0)

    for x in range(num_x):
        for a in range(num_a):
            for e in range(num_a):
                constraints.append(G[x, a, e] >> 0)

    objective = cp.Maximize(s)
    problem = cp.Problem(objective, constraints)

    problem.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
        },
    )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(
        f"Weight SDP failed: {problem.status}"
        )

    return 1.0 - float(s.value)


# ============================================================
# Fixed-assemblage guessing probability
# ============================================================

def solve_pguess_fixed_assemblage(assemblage, x_star):
    x_values = {x for x, a in assemblage.keys()}
    a_values = {a for x, a in assemblage.keys()}
    num_x = len(x_values)
    num_a = len(a_values)
    num_e = num_a

    dB = assemblage[0, 0].shape[0]

    I = np.eye(dB, dtype=complex)

    X = {}

    for x in range(num_x):
        for a in range(num_a):
            X[x, a] = cp.Variable((dB, dB), hermitian=True)

    Y = {}

    for x in range(num_x):
        for e in range(num_e):
            Y[x, e] = cp.Variable((dB, dB), hermitian=True)


    constraints = []

    for x in range(num_x):
        for a in range(num_a):
            for e in range(num_e):
                delta_x = int(x == x_star)
                delta_ae = int(a == e)
                constraints.append(
                    X[x,a] - Y[x,e] + delta_x * sum(Y[x_prime, e] for x_prime in range(num_x))
                    >> delta_x * delta_ae * I
                )

    objective = 0
    
    for x in range(num_x):
        for a in range(num_a):
            objective += cp.real(
                cp.trace(X[x, a] @ assemblage[x, a])
            )
    
    problem = cp.Problem(
        cp.Minimize(objective),
        constraints
    )

    problem.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
        },
    )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(
            f"Fixed-assemblage p_guess SDP failed: {problem.status}"
        )

    X_value = {}

    for x in range(num_x):
        for a in range(num_a):
            X_value[x, a] = X[x, a].value

    return float(problem.value), X_value

# ============================================================
# Exact minimum guessing probability
# ============================================================

def solve_pguess_exact(M, x_star):
    x_values = {x for x, a in M.keys()}
    a_values = {a for x, a in M.keys()}
    num_x = len(x_values)
    num_a = len(a_values)
    num_e = num_a

    dA = M[0, 0].shape[0]

    rho_A = cp.Variable((dA, dA), hermitian=True)

    X = {}

    for x in range(num_x):
        for a in range(num_a):
            X[x, a] = cp.Variable((dA, dA), hermitian=True)

    Y = {}

    for x in range(num_x):
        for e in range(num_e):
            Y[x, e] = cp.Variable((dA, dA), hermitian=True)

    # The constraints start here

    constraints = [
        rho_A >> 0,
        cp.real(cp.trace(rho_A)) == 1,
    ]

    for x in range(num_x):
        for a in range(num_a):
            for e in range(num_e):

                delta_x = int(x == x_star)
                delta_ae = int(a == e)

                sum_Y = sum(
                    Y[x_prime, e]
                    for x_prime in range(num_x)
                )

                constraints.append(
                    X[x, a] - Y[x, e] + delta_x * sum_Y
                    >> delta_x * delta_ae * rho_A
                )

    # The constraints end here

    objective = 0

    for x in range(num_x):
        for a in range(num_a):
            objective += cp.real(
                cp.trace(X[x, a] @ M[x, a])
            )

    problem = cp.Problem(
        cp.Minimize(objective),
        constraints
    )

    problem.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
        },
    )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(
            f"Exact p_guess SDP failed: {problem.status}"
        )

    return float(problem.value)


# ============================================================
# See-saw state update
# ============================================================

def best_state_for_dual(M, X_value):
    x_values = {x for x, a in M.keys()}
    a_values = {a for x, a in M.keys()}
    num_x = len(x_values)
    num_a = len(a_values)

    H = 0

    for x in range(num_x):
        for a in range(num_a):
            H += np.kron(M[x, a], X_value[x, a])

    H = (H + H.conj().T) / 2

    eigenvalues, eigenvectors = la.eigh(H)
    index = int(np.argmin(eigenvalues))
    psi = eigenvectors[:, index]
    psi = psi / la.norm(psi)

    return np.outer(psi, psi.conj())


# ============================================================
# See-saw minimization
# ============================================================

def seesaw_minimize_pguess(
    M,
    x_star,
    dB,
    rho_init=None,
    max_iter=4,
    n_runs=2,
    seed=0,
    tol_p=1e-8,
    tol_rho=1e-10,
    patience=2,
):
    dA = M[0, 0].shape[0]
    dAB = dA * dB
    rng = np.random.default_rng(seed)

    best_p = np.inf
    best_rho_AB = None

    def run_from(rho_AB):
        p_prev = None
        rho_prev = None
        stable_count = 0

        for _ in range(max_iter):
            assemblage = assemblage_from_state(M, rho_AB)

            p_guess, X_value = solve_pguess_fixed_assemblage(
                assemblage,
                x_star,
            )

            rho_next = best_state_for_dual(M, X_value)

            if p_prev is not None:
                dp = abs(p_guess - p_prev)
                drho = np.linalg.norm(
                    rho_next - rho_prev,
                    ord="fro",
                )

                if dp < tol_p and drho < tol_rho:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= patience:
                    rho_AB = rho_next
                    break

            p_prev = p_guess
            rho_prev = rho_next
            rho_AB = rho_next

        assemblage = assemblage_from_state(M, rho_AB)

        p_final, _ = solve_pguess_fixed_assemblage(
            assemblage,
            x_star,
        )

        return float(p_final), rho_AB

    restarts_done = 0

    if rho_init is not None:
        p_guess, rho_AB = run_from(rho_init)
        best_p = p_guess
        best_rho_AB = rho_AB
        restarts_done = 1

    for _ in range(restarts_done, n_runs):
        psi = rng.normal(size=dAB) + 1j * rng.normal(size=dAB)
        psi = psi / la.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        p_guess, rho_AB = run_from(rho0)

        if p_guess < best_p:
            best_p = p_guess
            best_rho_AB = rho_AB

    return float(best_p), best_rho_AB


# ============================================================
# plot
# ============================================================

def run_plot():
    x_star = 0
    dB = 5

    etas = np.linspace(0.65, 1.0, 351)

    weight_list = []
    p_exact_list = []
    p_seesaw_list = []

    rho_previous = None

    for i, eta in enumerate(etas):
        M = povms_eta(float(eta))

        dA = M[0, 0].shape[0]

        a_values = {a for x, a in M.keys()}
        num_a = len(a_values)

        if dB < dA:
            raise ValueError(
                "Theorem 1 requires dB >= dA for equality."
            )

        weight = solve_weight(M, x_star)
        p_exact = solve_pguess_exact(M, x_star)

        p_seesaw, rho_seesaw = seesaw_minimize_pguess(
            M,
            x_star=x_star,
            dB=dB,
            rho_init=rho_previous,
            max_iter=4,
            n_runs=2,
            seed=i,
            tol_p=1e-8,
            tol_rho=1e-10,
            patience=2,
        )

        rho_previous = rho_seesaw

        weight_list.append(weight)
        p_exact_list.append(p_exact)
        p_seesaw_list.append(p_seesaw)

        factor = num_a / (num_a - 1)
        weight_lb_exact = factor * (1 - p_exact)

        print(
            f"eta={eta:.3f} | "
            f"W={weight:.8f} | "
            f"p_exact={p_exact:.8f} | "
            f"p_seesaw={p_seesaw:.8f} | "
        )

    weight = np.array(weight_list)
    p_exact = np.array(p_exact_list)
    p_seesaw = np.array(p_seesaw_list)

    factor = num_a / (num_a - 1)
    weight_lb_exact = factor * (1 - p_exact)
    weight_lb_seesaw = factor * (1 - p_seesaw)

    fig, ax = plt.subplots()

    ax.plot(
        etas,
        weight,
        linestyle="--",
        linewidth=1.5,
        label=r"$W^{1}(\mathbb{M}^{\mathcal{A}|\mathcal{I}}_{\mathrm{A}})$",
    )

    ax.plot(
        etas,
        weight_lb_exact,
        linestyle="-",
        linewidth=1.5,
        label=r"$2(1-p^{1}_{\rm guess,exact})$",
    )

    ax.plot(
        etas,
        weight_lb_seesaw,
        linestyle=":",
        linewidth=1.5,
        label=r"$2(1-p^{1}_{\rm guess,seesaw})$",
    )

    ax.set_xlabel(r"$\eta$", fontsize=15)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle=":")
    ax.tick_params(axis="both", labelsize=15)

    axins = inset_axes(
        ax,
        width="35%",
        height="30%",
        loc="upper left",
        bbox_to_anchor=(0.15, -0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    axins.plot(etas, weight, linestyle="--", linewidth=1.5)
    axins.plot(etas, weight_lb_exact, linestyle="-", linewidth=1.5)
    axins.plot(etas, weight_lb_seesaw, linestyle=":", linewidth=1.5)

    axins.set_xlim(0.705, 0.710)
    axins.set_ylim(0.0, 0.04)

    axins.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
    axins.xaxis.set_minor_locator(mticker.MultipleLocator(0.001))
    axins.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    axins.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    axins.grid(True, which="major", linestyle="-")
    axins.grid(True, which="minor", linestyle=":")
    axins.tick_params(axis="both", labelsize=11)

    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.4",
    )

    ax.legend(
        fontsize=15,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.72,
        bottom=0.12,
        top=0.95,
    )

    plt.show()


if __name__ == "__main__":
    run_plot()
