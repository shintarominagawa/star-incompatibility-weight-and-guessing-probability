import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cvxpy as cp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =========================
# Basic quantum utilities
# =========================
def paulis():
    I = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, sx, sy, sz

def povms_eta(eta: float):
    I, sx, sy, sz = paulis()

    def bin_povm(sigma):
        return [(I + eta * sigma) / 2.0, (I - eta * sigma) / 2.0]

    return [bin_povm(sx), bin_povm(sy), bin_povm(sz)]

def project_to_density_matrix(rho: np.ndarray):
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        raise ValueError("Trace too small.")
    rho = rho / tr
    w, v = la.eigh(rho)
    w = np.maximum(w, 0.0)
    rho = (v * w) @ v.conj().T
    rho = rho / np.trace(rho)
    return rho

def partial_trace_B(rho_AB: np.ndarray, dA: int, dB: int):
    rho4 = rho_AB.reshape(dA, dB, dA, dB)
    rhoA = np.zeros((dA, dA), dtype=complex)
    for i in range(dA):
        for j in range(dA):
            for b in range(dB):
                rhoA[i, j] += rho4[i, b, j, b]
    rhoA = (rhoA+rhoA.conj().T)/2.0
    return rhoA

def partial_trace_A(rho_AB: np.ndarray, dA: int, dB: int):
    rho4 = rho_AB.reshape(dA, dB, dA, dB)  # (a, i, a', j)
    rhoB = np.zeros((dB, dB), dtype=complex)
    for i in range(dB):
        for j in range(dB):
            for a in range(dA):
                rhoB[i, j] += rho4[a, i, a, j]
    rhoB = (rhoB+rhoB.conj().T)/2.0
    return rhoB

def assemblage_from_state(M, rho_AB: np.ndarray):
    dA = M[0][0].shape[0]
    dAB = rho_AB.shape[0]
    dB = dAB // dA

    I_B = np.eye(dB, dtype=complex)

    sigma = []
    for x in range(len(M)):
        sigma.append([]) #sigma = [ [], [], ... ] 
        for a in range(len(M[0])):
            tmp = np.kron(M[x][a], I_B) @ rho_AB
            s = partial_trace_A(tmp, dA, dB)
            sigma[x].append(s)
    return sigma

# SDP variable
def hermitian_var(d: int, name: str):
    return cp.Variable((d, d), hermitian=True, name=name)

# =========================
# Solver configuration 
# =========================
def get_solver_cfg(solve_cfg: dict):
    solver = solve_cfg["solver"].upper()
    verbose = bool(solve_cfg.get("verbose", False))
    opts = dict(solve_cfg.get("opts", {}))
    return solver, verbose, opts

# =========================
# Star-incompatibility Weight 
# =========================
def solve_weight(M, x_star: int, solve_cfg: dict):

    solver, verbose, solver_opts = get_solver_cfg(solve_cfg)

    d = M[0][0].shape[0]
    num_x = len(M)
    num_a = len(M[0])  # assume all POVMs have same number of outcomes
    I = np.eye(d, dtype=complex)

    s = cp.Variable(name="s")
    constraints = [s >= 0, s <= 1]

    Ftilde = []
    for x in range(num_x):
        Ftilde.append([])
        for a in range(num_a):
            Ftilde[x].append(hermitian_var(d, f"Ftilde_{x}_{a}"))
            constraints.append(Ftilde[x][a] >> 0)

    Gtilde = []
    for x in range(num_x):
        Gtilde.append([])
        for a in range(num_a):
            Gtilde[x].append([])
            for e in range(num_a):
                Gtilde[x][a].append(hermitian_var(d, f"Gtilde_{x}_{a}_{e}"))
                constraints.append(Gtilde[x][a][e] >> 0)

    for x in range(num_x):
        constraints.append(cp.sum(Ftilde[x]) == s * I)

    for x in range(num_x):
        for a in range(num_a):
            constraints.append(cp.sum(Gtilde[x][a]) == Ftilde[x][a])

    for x in range(num_x):
        for e in range(num_a):
            sum_over_a = 0
            for a in range(num_a):
                sum_over_a += Gtilde[x][a][e]
            constraints.append(sum_over_a == Ftilde[x_star][e])

    for x in range(num_x):
        for a in range(num_a):
            constraints.append(M[x][a] - Ftilde[x][a] >> 0)

    prob = cp.Problem(cp.Maximize(s), constraints)
    prob.solve(solver=solver, verbose=verbose, **solver_opts)

    s_opt = float(s.value) if s.value is not None else np.nan
    s_opt = min(max(s_opt, 0.0), 1.0)
    W = 1.0 - s_opt
    return W

# =========================
# star-incompatibility weight dual
# =========================
def solve_weight_dual(M, x_star: int, solve_cfg: dict):

    solver, verbose, solver_opts = get_solver_cfg(solve_cfg)

    d = M[0][0].shape[0]
    num_x = len(M)
    num_a = len(M[0])

    # Variables
    R = [[hermitian_var(d, f"R_{x}_{a}") for a in range(num_a)] for x in range(num_x)]
    X = [[hermitian_var(d, f"X_{x}_{a}") for a in range(num_a)] for x in range(num_x)]
    Y = [[hermitian_var(d, f"Y_{e}_{x}") for x in range(num_x)] for e in range(num_a)]
    Z = [hermitian_var(d, f"Z_{x}") for x in range(num_x)]

    constraints = []

    constraints.append(cp.sum([cp.real(cp.trace(Z[x])) for x in range(num_x)]) == 1.0)

    sumY_over_x = []
    for a in range(num_a):
        sumY_over_x.append(cp.sum([Y[a][xp] for xp in range(num_x)]))

    for x in range(num_x):
        for a in range(num_a):
            if x == x_star:
                constraints.append(R[x][a] == X[x][a] + Z[x] + sumY_over_x[a])
            else:
                constraints.append(R[x][a] == X[x][a] + Z[x])

    for x in range(num_x):
        for a in range(num_a):
            for e in range(num_a):
                constraints.append(X[x][a] + Y[e][x] >> 0)

    for x in range(num_x):
        for a in range(num_a):
            constraints.append(R[x][a] >> 0)

    obj_terms = []
    for x in range(num_x):
        for a in range(num_a):
            obj_terms.append(cp.real(cp.trace(R[x][a] @ M[x][a])))
    objective = cp.Minimize(cp.sum(obj_terms))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose, **solver_opts)

    opt = float(prob.value) if prob.value is not None else np.nan
    dual_W = 1.0-opt

    return dual_W

# =========================
# guessing probability dual
# =========================
def solve_pguess_dual(assemblage, x_star: int, solve_cfg: dict):
    
    solver, verbose, solver_opts = get_solver_cfg(solve_cfg)

    num_x = len(assemblage)
    num_a = len(assemblage[0])
    num_e = num_a
    dB = assemblage[0][0].shape[0]

    I_B = np.eye(dB, dtype=complex)
    Z_B = np.zeros((dB, dB), dtype=complex)

    F = []
    for x in range(num_x):
        F.append([])
        for a in range(num_a):
            F[x].append(hermitian_var(dB, f"F_{x}_{a}"))

    G_vars = {}
    for x in range(num_x):
        if x == x_star:
            continue
        G_vars[x] = []
        for e in range(num_e):
            G_vars[x].append(hermitian_var(dB, f"G_{x}_{e}"))

    constraints = []

    for x in range(num_x):
        if x == x_star:
            continue
        for a in range(num_a):
            for e in range(num_e):
                constraints.append(F[x][a] - G_vars[x][e] >> 0)

    for a in range(num_a):
        for e in range(num_e):
            sumG = 0
            for x in range(num_x):
                if x == x_star:
                    continue
                sumG += G_vars[x][e]

            rhs = I_B if a == e else Z_B
            constraints.append(F[x_star][a] + sumG >> rhs)

    objective_terms = []
    for x in range(num_x):
        for a in range(num_a):
            objective_terms.append(cp.real(cp.trace(F[x][a] @ assemblage[x][a])))

    obj = cp.Minimize(cp.sum(objective_terms))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose, **solver_opts)

    p_guess = float(prob.value)

    F_val = []
    for x in range(num_x):
        F_val.append([])
        for a in range(num_a):
            F_val[x].append(F[x][a].value)

    return p_guess, F_val, prob.status

def find_best_state(M, F_val):
    num_x = len(M)
    num_a = len(M[0])

    H = 0
    for x in range(num_x):
        for a in range(num_a):
            H += np.kron(M[x][a], F_val[x][a])

    H = (H + H.conj().T) / 2.0
    w, v = la.eigh(H)
    psi = v[:, int(np.argmin(w))]
    psi = psi / la.norm(psi)
    rho = np.outer(psi, psi.conj())
    return project_to_density_matrix(rho)

# =========================
# See-saw optimization
# =========================

def seesaw_minimize_pguess(M, x_star: int, dB: int, rho_init, seesaw_cfg: dict, solve_cfg: dict):
    """
    seesaw_cfg keys:
      - max_iter: maximum iterations per restart
      - n_restart: number of restarts
      - seed: base random seed
      - tol_p: convergence tolerance for p_guess change
      - tol_rho: convergence tolerance for rho change
      - patience: require 'patience' consecutive small changes to stop
    """
    max_iter = int(seesaw_cfg["max_iter"])
    n_restart = int(seesaw_cfg["n_restart"])
    seed = int(seesaw_cfg["seed"])
    tol_p = float(seesaw_cfg["tol_p"])
    tol_rho = float(seesaw_cfg["tol_rho"])
    patience = int(seesaw_cfg["patience"])

    dA = M[0][0].shape[0]
    dAB = dA * dB
    rng = np.random.default_rng(seed)

    best_p = np.inf
    best_rho_AB = None

    def run_from(rho_AB):
        rho_AB = project_to_density_matrix(rho_AB)

        p_prev = None
        rho_prev = None
        stable_count = 0

        for it in range(max_iter):
            sigma = assemblage_from_state(M, rho_AB)
            p, F_val, _ = solve_pguess_dual(sigma, x_star=x_star, solve_cfg=solve_cfg)
            rho_next = find_best_state(M, F_val)

            # convergence check starts from 2nd iteration
            if p_prev is not None:
                dp = abs(float(p) - float(p_prev))
                dr = np.linalg.norm(rho_next - rho_prev, ord="fro")

                if (dp < tol_p) and (dr < tol_rho):
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= patience:
                    rho_AB = rho_next
                    break

            p_prev = float(p)
            rho_prev = rho_next
            rho_AB = rho_next

        # final eval for consistent reporting
        sigma = assemblage_from_state(M, rho_AB)
        p_final, _, _ = solve_pguess_dual(sigma, x_star=x_star, solve_cfg=solve_cfg)
        return float(p_final), project_to_density_matrix(rho_AB)

    restarts_done = 0
    if rho_init is not None:
        p, rho = run_from(rho_init)
        best_p, best_rho_AB = p, rho
        restarts_done = 1

    for _ in range(restarts_done, n_restart):
        vec = rng.normal(size=dAB) + 1j * rng.normal(size=dAB)
        vec = vec / la.norm(vec)
        rho0 = np.outer(vec, vec.conj())
        p, rho = run_from(rho0)
        if p < best_p:
            best_p, best_rho_AB = p, rho

    rho_A = partial_trace_B(best_rho_AB, dA=dA, dB=dB)
    return float(best_p), best_rho_AB, rho_A

# =========================
# run, sweep, and plot
# =========================

def run_sweep_plot(sweep_cfg: dict, seesaw_cfg: dict, solve_cfg: dict):
    n_eta = int(sweep_cfg["n_eta"])
    x_star = int(sweep_cfg["x_star"])
    dB = int(sweep_cfg["dB"])

    # 0\le \eta\le 1
    etas = np.linspace(0.65, 1.0, n_eta)

    W_list = []
    Wlb_list = []
    gap_list = []
    pguess_list = []

    rho_prev = None

    for i, eta in enumerate(etas):
        M = povms_eta(float(eta))

        m = len(M[x_star])  # |A|
        if m <= 1:
            raise ValueError("Need at least 2 outcomes to define the bound.")

        W = solve_weight(M, x_star=x_star, solve_cfg=solve_cfg)
        W_list.append(W)

        W_dual = solve_weight_dual(M, x_star=x_star, solve_cfg=solve_cfg)

        seesaw_cfg_eta = dict(seesaw_cfg)
        seesaw_cfg_eta["seed"] = int(seesaw_cfg["seed"]) + i

        p_guess, rho_AB_star, _ = seesaw_minimize_pguess(
            M, x_star=x_star, dB=dB, rho_init=rho_prev,
            seesaw_cfg=seesaw_cfg_eta, solve_cfg=solve_cfg
        )
        rho_prev = rho_AB_star

        p_guess = min(max(float(p_guess), 0.0), 1.0)
        pguess_list.append(p_guess)

        W_lb = (m * (1.0 - p_guess)) / (m - 1.0)
        Wlb_list.append(W_lb)

        gap = W - W_lb
        gap_list.append(gap)

        flag = "OK" if gap >= -1e-8 else "VIOLATION?"
        print(f"eta={eta:.4f} | W={W:.6f} | W_dual={W_dual:.6f} | p_guess={p_guess:.6f} | W_LB={W_lb:.6f} | gap={gap:.3e} | {flag}")

        # --- main plot ---
    fig, ax = plt.subplots()

    #ax.plot(etas, pguess_list, linestyle=':', linewidth=2,
            #label=r"$p^{1}_{\rm guess}(\mathbb{M}^{\mathcal{A}|\mathcal{I}}_\mathrm{A},\rho_{\mathrm{AB}})$")
    ax.plot(etas, W_list, linestyle='--', linewidth=1.5, label=r"$W^{1}(\mathbb{M}^{\mathcal{A}|\mathcal{I}}_\mathrm{A})$")
    ax.plot(etas, Wlb_list, linestyle='-', linewidth=1.5, label=r"$2(1-p^{1}_{\rm guess})$")

    ax.set_xlabel(r"$\eta$", fontsize=15)
    ax.grid(True)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    ax.grid(True, which="major", linestyle='-')
    ax.grid(True, which="minor", linestyle=':')

    ax.tick_params(axis="both", labelsize=15)

    # --- inset settings ---
    # inset axes: place inside the main axes
    axins = inset_axes(
        ax,
        width="35%", height="30%",  # size of inset
        loc="upper left",           # position; try "upper right" etc.
        bbox_to_anchor=(0.15, -0.1, 1, 1),   # (x0, y0, w, h)
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    # plot the same three curves on inset
    axins.plot(etas, W_list, linestyle='--', linewidth=1.5)
    axins.plot(etas, Wlb_list, linestyle='-', linewidth=1.5)

    # inset range
    axins.set_xlim(0.70, 0.715)
    axins.set_ylim(0.0, 0.04)

    axins.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    axins.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    axins.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
    axins.xaxis.set_minor_locator(mticker.MultipleLocator(0.001))

    axins.grid(True, which="major", linestyle='-')
    axins.grid(True, which="minor", linestyle=':')
    axins.tick_params(axis="both", labelsize=11)

    # mark the zoomed region on the main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

    # legend outside
    ax.legend(
        fontsize=15,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    plt.show()

    return etas, np.array(W_list), np.array(Wlb_list), np.array(gap_list), np.array(pguess_list)

# =========================
# Main settings
# =========================

if __name__ == "__main__":
    solve_cfg = {
        "solver": "MOSEK", 
        "verbose": False,
        "opts": {
            "mosek_params": {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
            }
        }
    }
    #SCS
    # solve_cfg_scs = {
    #     "solver": "SCS",
    #     "verbose": False,
    #     "opts": {
    #         "eps": 1e-8,          
    #         "max_iters": 200000, 
    #         "alpha": 1.5,       
    #         "scale": 1.0,       
    #         "normalize": True,   
    #         "use_indirect": False  
    #     }
    # }

    sweep_cfg = {
        "n_eta": 351,
        "x_star": 0,
        "dB": 2,
    }

    seesaw_cfg = {
        "max_iter": 100,
        "n_restart": 10,
        "seed": 0,
        "tol_p": 1e-8,
        "tol_rho": 1e-10,
        "patience": 4,
    }

    run_sweep_plot(sweep_cfg=sweep_cfg, seesaw_cfg=seesaw_cfg, solve_cfg=solve_cfg)