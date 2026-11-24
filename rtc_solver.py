"""
rtc_solver.py -- special-relativistic Taylor-Couette flow between coaxial cylinders.

Units: c = 1, eta = 1, wall temperature scale T_w = 1, inner radius R1 = 1, so
that kappa = lambda = kappa T_w/(eta c^2). The cylinders rotate with Omega1, Omega2.

Model 1, zero heat flux (q = 0), angular-momentum integral only:
    eta r^3 gamma^3 Omega' = -K_L0,          gamma = (1 - r^2 Omega^2)^(-1/2).
    Shooting inwards from the outer wall (DOP853) with Brent's method on K_L0.

Model 2, Eckart heat conduction, first integrals r T^{rt} = K_E, r^3 T^{r phi} = K_L:
    eta r^3 gamma Omega'             = r^2 Omega K_E - K_L,
    Theta' - r gamma^2 Omega^2 Theta = -(gamma/r)(K_E - Omega K_L),
    Theta = kappa T/(eta c^2). Unknowns: rapidity y = artanh(r Omega), Theta, and
    M(r) = int_{R1}^r r' gamma^2 Omega^2 dr', so that mu = exp(-M) is the
    integrating factor of the temperature equation. Collocation (solve_bvp) with
    (K_L, K_E) as unknown parameters and continuation in the wall rapidities;
    cross-checked by two-parameter shooting.

Wall-temperature protocols (T_w = 1):
    'equal'    T(R1) = T(R2) = 1
    'tolman'   T(R_i) = gamma_i              (equal T/gamma at the walls)
    'balanced' T(R1) = 1, T(R2) = 1/mu(R2)   (imposed self-consistently)
    lambda = 0 means the limit kappa -> 0 at fixed wall temperatures: Theta(R_i) = 0.
For balanced walls the solution is the lambda -> 0 solution for every lambda,
because the conductivity term of the exact energy-flux relation vanishes.

Usage:
    python rtc_solver.py            run every stage that is not cached yet
    python rtc_solver.py --force    recompute everything
    python rtc_solver.py pn planar  run only the named stages
Results: results/*.json and results/summary.txt. Then run make_figures.py.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.integrate import quad, simpson, solve_bvp, solve_ivp
from scipy.optimize import brentq, fsolve

ETA = 1.0
RTOL, ATOL = 1e-12, 1e-14            # ODE tolerances (shooting)
BVP_TOL = 1e-9                        # collocation tolerance
R1, R2 = 1.0, 2.0                     # default geometry


# ----------------------------------------------------------------------------
# Classical baseline and O(v^2/c^2) theory
# ----------------------------------------------------------------------------
def classical(R1, R2, O1, O2):
    """A, B and K_cl = 2 eta B of the classical profile Omega = A + B/r^2."""
    A = (O2 * R2**2 - O1 * R1**2) / (R2**2 - R1**2)
    B = R1**2 * R2**2 * (O1 - O2) / (R2**2 - R1**2)
    return A, B, 2.0 * ETA * B


def pn_corrections(R1, R2, O1, O2, lam):
    """Leading relativistic torque corrections R - 1 at the given speeds.

    zero_flux  : (3/2) I2/I3
    heat_lam0  : (I1^2/l + I2/2)/I3      (lambda -> 0, and balanced walls)
    heat_equal : heat_lam0 + lambda I1 delta/(l K_cl I3), delta = I_T
    heat_tolman: heat_lam0 + lambda I1 delta/(l K_cl I3), delta = I_T + gamma1 - gamma2
    delta is the wall-temperature mismatch T1 - mu(R2) T2 to leading order."""
    A, B, Kcl = classical(R1, R2, O1, O2)
    Om = lambda r: A + B / r**2
    I1 = quad(lambda r: Om(r) / r, R1, R2, epsabs=1e-14)[0]
    I2 = quad(lambda r: Om(r)**2 / r, R1, R2, epsabs=1e-14)[0]
    I3 = quad(lambda r: r**-3, R1, R2, epsabs=1e-14)[0]
    IT = quad(lambda r: r * Om(r)**2, R1, R2, epsabs=1e-14)[0]
    ell = np.log(R2 / R1)
    dg = 0.5 * (R1 * O1)**2 - 0.5 * (R2 * O2)**2          # gamma1 - gamma2 to O(v^2)
    base = (I1**2 / ell + 0.5 * I2) / I3
    k = lam * I1 / (ell * Kcl * I3)
    return dict(I1=I1, I2=I2, I3=I3, IT=IT, ell=ell, zero_flux=1.5 * I2 / I3,
                heat_lam0=base, heat_equal=base + k * IT, heat_tolman=base + k * (IT + dg))


# ----------------------------------------------------------------------------
# Model 1: zero heat flux
# ----------------------------------------------------------------------------
def _zf_integrate(K, R1, R2, O2, dense=False):
    """Integrate eta r^3 gamma^3 Omega' = -K inwards from R2."""
    def rhs(r, y):
        s = max(1.0 - (r * y[0])**2, 0.0)
        return [-K / (ETA * r**3) * s**1.5]

    def light_cylinder(r, y):
        return 1.0 - 1e-13 - abs(r * y[0])
    light_cylinder.terminal = True
    return solve_ivp(rhs, [R2, R1], [O2], method='DOP853', rtol=RTOL,
                     atol=ATOL, events=light_cylinder, dense_output=dense)


def solve_zero_flux(R1, R2, O1, O2, npts=801):
    """Zero-heat-flux solution: dict(K, R = K/K_cl, r, Omega, bc_error)."""
    _, _, Kcl = classical(R1, R2, O1, O2)

    def mismatch(K):                       # increases monotonically with K
        s = _zf_integrate(K, R1, R2, O2)
        if s.status == 1:                  # reached the light cylinder
            return np.sign(s.y[0, -1]) * 10.0
        return s.y[0, -1] - O1

    a, b = Kcl - 0.5 * abs(Kcl), Kcl + 0.5 * abs(Kcl)
    fa, fb = mismatch(a), mismatch(b)
    while fa > 0:
        a = a - (b - a)
        fa = mismatch(a)
    while fb < 0:
        b = b + 2 * (b - a)
        fb = mismatch(b)
    K = brentq(mismatch, a, b, xtol=1e-15, rtol=1e-15, maxiter=500)
    s = _zf_integrate(K, R1, R2, O2, dense=True)
    r = np.linspace(R1, R2, npts)
    return dict(K=K, R=K / Kcl, r=r, Omega=s.sol(r)[0], bc_error=abs(s.y[0, -1] - O1))


# ----------------------------------------------------------------------------
# Model 2: Eckart heat conduction
# ----------------------------------------------------------------------------
def _heat_rhs(inertial=True):
    def f(r, Y, p):
        y, Th, M = Y
        KL, KE = p
        c, s, th = np.cosh(y), np.sinh(y), np.tanh(y)
        Om = th / r
        acc = r * c**2 * Om**2                      # = r gamma^2 Omega^2
        dy = s * c / r + c * (r * KE * th - KL) / (ETA * r**2)
        dTh = (acc * Th if inertial else 0.0) - c * (KE - Om * KL) / r
        return np.vstack([dy, dTh, acc])
    return f


def wall_temperatures(protocol, R1, R2, O1, O2):
    """T(R1), T(R2) in units of T_w ('balanced': T(R2) is fixed by mu(R2))."""
    g1, g2 = 1 / np.sqrt(1 - (R1 * O1)**2), 1 / np.sqrt(1 - (R2 * O2)**2)
    if protocol == 'equal':
        return 1.0, 1.0
    if protocol == 'tolman':
        return g1, g2
    if protocol == 'balanced':
        return 1.0, None
    raise ValueError(protocol)


def solve_heat(R1, R2, O1, O2, lam, protocol='equal', inertial=True,
               nsteps=None, record=False, nodes=200, tol=None):
    """Heat-conducting solution by continuation in the wall rapidities.

    The wall rapidities are scaled by s, from s = 0.02 (classical initial
    guess) to s = 1. With record=True the list (O1, O2, K_L, K_E) along the
    path is returned in out['path']."""
    tol = BVP_TOL if tol is None else tol
    y1t, y2t = np.arctanh(R1 * O1), np.arctanh(R2 * O2)
    if nsteps is None:
        nsteps = int(40 + 60 * max(abs(y1t), abs(y2t)))
    f = _heat_rhs(inertial)
    path, sol = [], None
    for s in np.linspace(0.02, 1.0, nsteps):
        o1, o2 = np.tanh(s * y1t) / R1, np.tanh(s * y2t) / R2
        T1, T2 = wall_temperatures(protocol, R1, R2, o1, o2)
        if sol is None:
            A, B, Kcl = classical(R1, R2, o1, o2)
            x = np.linspace(R1, R2, nodes)
            t2 = T1 if T2 is None else T2
            Y = np.vstack([np.arctanh(x * (A + B / x**2)),
                           lam * (T1 + (t2 - T1) * (x - R1) / (R2 - R1)), np.zeros_like(x)])
            p = np.array([Kcl, 0.0])
        else:
            x, Y, p = sol.x, sol.y, sol.p

        def bc(a, b, p, o1=o1, o2=o2, T1=T1, T2=T2):
            th2 = lam * (np.exp(b[2]) * T1 if T2 is None else T2)
            return np.array([a[0] - np.arctanh(R1 * o1), b[0] - np.arctanh(R2 * o2),
                             a[1] - lam * T1, b[1] - th2, a[2]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            new = solve_bvp(f, bc, x, Y, p=p, tol=tol, bc_tol=1e-12, max_nodes=500000)
        if new.status != 0:
            raise RuntimeError(f'collocation failed at O1={o1:.5f}: {new.message}')
        sol = new
        if record:
            path.append((o1, o2, sol.p[0], sol.p[1]))
    return _package(sol, R1, R2, O1, O2, lam, protocol, inertial, path)


def _package(sol, R1, R2, O1, O2, lam, protocol, inertial, path, npts=4001):
    """Derived fields and accuracy checks of a collocation solution."""
    A, B, Kcl = classical(R1, R2, O1, O2)
    KL, KE = sol.p
    r = np.linspace(R1, R2, npts)
    y, Th, M = sol.sol(r)
    yp, Thp, _ = sol.sol.derivative()(r)
    g = np.cosh(y)
    Om = np.tanh(y) / r
    Omp = yp / (g**2 * r) - np.tanh(y) / r**2
    acc = r * g**2 * Om**2
    q = -(Thp - (acc * Th if inertial else 0.0))          # Eckart's law, eta c^2 units
    mu = np.exp(-M)
    # (i) first integrals on a dense grid between the collocation nodes
    Trphi = -ETA * g**3 * Omp + g * Om * q
    Trt = -ETA * r**2 * g**3 * Om * Omp + g * q
    res_L = np.max(np.abs(r**3 * Trphi - KL)) / abs(KL)
    res_E = np.max(np.abs(r * Trt - KE)) / max(abs(KE), 1e-300)
    # (ii) global energy budget (Omega1 - Omega2) K_L = R2 q2/gamma2 - R1 q1/gamma1
    b_rhs = R2 * q[-1] / g[-1] - R1 * q[0] / g[0]
    budget_rel = abs((O1 - O2) * KL - b_rhs) / abs((O1 - O2) * KL)
    # (iii) local energy balance in integrated form [mu r q] = int mu Phi r dr
    Phi = ETA * r**2 * g**4 * Omp**2
    if inertial:
        energy_rel = abs((mu[-1] * r[-1] * q[-1] - r[0] * q[0]) / simpson(mu * Phi * r, x=r) - 1)
    else:
        energy_rel = np.nan
    # (iv) exact energy-flux relation
    m = mu if inertial else np.ones_like(r)
    lhs = KE * simpson(m * g / r, x=r)
    rhs = KL * simpson(m * g * Om / r, x=r) + (Th[0] - m[-1] * Th[-1])
    out = dict(R1=R1, R2=R2, O1=O1, O2=O2, lam=lam, protocol=protocol,
               inertial=inertial, KL=float(KL), KE=float(KE), R=float(KL / Kcl),
               Kcl=Kcl, r=r, Omega=Om, Theta=Th, q=q, Phi=Phi, mu=mu, mu2=float(mu[-1]),
               res_L=float(res_L), res_E=float(res_E), budget_rel=float(budget_rel),
               energy_rel=float(energy_rel),
               KE_relation_rel=float(abs(lhs - rhs) / abs(lhs)),
               dOmega_max=float(np.max(np.abs(Om - (A + B / r**2)))),
               Theta_min_interior=float(Th[1:-1].min()), nodes=len(sol.x), path=path)
    if lam > 0:
        T = Th / lam
        out.update(T=T, T1=float(T[0]), T2=float(T[-1]), T_min=float(T.min()),
                   dT_max=float(T.max() - max(T[0], T[-1])), r_Tmax=float(r[np.argmax(T)]),
                   delta=float(T[0] - mu[-1] * T[-1]))
    return out


def shoot_heat(R1, R2, O1, O2, lam, protocol, guess):
    """Independent check: two-parameter shooting from the inner wall
    (MINPACK hybrid method on (K_L, K_E))."""
    f = _heat_rhs(True)
    T1, T2 = wall_temperatures(protocol, R1, R2, O1, O2)

    def mismatch(p):
        s = solve_ivp(lambda r, Y: f(r, Y.reshape(3, 1), p).ravel(), [R1, R2],
                      [np.arctanh(R1 * O1), lam * T1, 0.0], method='DOP853',
                      rtol=RTOL, atol=ATOL)
        th2 = lam * (np.exp(s.y[2, -1]) * T1 if T2 is None else T2)
        return [s.y[0, -1] - np.arctanh(R2 * O2), s.y[1, -1] - th2]

    p = fsolve(mismatch, guess, xtol=1e-13)
    return p, float(np.max(np.abs(mismatch(p))))


# ----------------------------------------------------------------------------
# Planar reference solutions for the narrow-gap check
# ----------------------------------------------------------------------------
def planar_heat_profile(x, d, v1):
    """Planar Eckart solution, equal plate temperatures, plate at x = 0 moving
    with v1 and plate at x = d at rest. In the frame where the plates move
    with +vp and -vp, v = sin[(1 - 2x/d) arcsin vp] (equivalently
    gamma v = tan[...], Gavassino et al. 2026); the profile is then boosted."""
    vp = np.tanh(0.5 * np.arctanh(v1))
    vs = np.sin((1 - 2 * x / d) * np.arcsin(vp))
    return (vs + vp) / (1 + vs * vp)


def planar_zero_flux_profile(x, d, v1):
    """Narrow-gap limit of the zero-heat-flux model: gamma v linear in x."""
    u1 = v1 / np.sqrt(1 - v1**2)
    u = u1 * (1 - x / d)
    return u / np.sqrt(1 + u**2)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
CASES = [(0.50, 0.20), (0.00, 0.20), (0.50, 0.00), (0.90, 0.00)]
HEAT = [(0.0, 'equal', 'lam0'), (1.0, 'equal', 'lam1_equal'),
        (1.0, 'tolman', 'lam1_tolman'), (1.0, 'balanced', 'lam1_balanced')]
KEYS = ('R', 'KL', 'KE', 'dOmega_max', 'res_L', 'res_E', 'budget_rel', 'energy_rel',
        'KE_relation_rel', 'Theta_min_interior', 'mu2', 'T1', 'T2', 'T_min', 'dT_max',
        'r_Tmax', 'delta')


def rayleigh_stable(O1, O2):
    """Classical Rayleigh criterion: (r^2 Omega)^2 increases outwards."""
    return (R2**2 * O2)**2 > (R1**2 * O1)**2


def stage_cases(say):
    out = {}
    for O1, O2 in CASES:
        zf = solve_zero_flux(R1, R2, O1, O2)
        A, B, Kcl = classical(R1, R2, O1, O2)
        e = dict(O1=O1, O2=O2, Kcl=Kcl, rayleigh_stable=bool(rayleigh_stable(O1, O2)),
                 Br_lam1=max(R1 * abs(O1), R2 * abs(O2))**2,
                 zero_flux=dict(R=zf['R'], K=zf['K'],
                                dOmega=float(np.max(np.abs(zf['Omega'] - A - B / zf['r']**2)))))
        say(f'({O1:.2f},{O2:.2f}) K_cl={Kcl:.6f} Rayleigh-stable={e["rayleigh_stable"]}')
        say(f'    zero flux      R0={zf["R"]:.6f}  max|dOmega|={e["zero_flux"]["dOmega"]:.6f}')
        for lam, prot, tag in HEAT:
            h = solve_heat(R1, R2, O1, O2, lam, prot)
            e[tag] = {k: h.get(k) for k in KEYS}
            say(f'    {tag:14s} R={h["R"]:.6f} K_E={h["KE"]:+.6f} | res {h["res_L"]:.0e} '
                f'{h["res_E"]:.0e} budget {h["budget_rel"]:.0e} energy {h["energy_rel"]:.0e} '
                f'KE-rel {h["KE_relation_rel"]:.0e} minTheta {h["Theta_min_interior"]:.1e}'
                + (f' | T1={h["T1"]:.4f} T2={h["T2"]:.4f} dT_max={h["dT_max"]:.4f} '
                   f'delta={h["delta"]:+.4f}' if lam > 0 else ''))
        # sign rule: R(lambda) - R(0) has the sign of delta * K_cl
        for tag in ('lam1_equal', 'lam1_tolman'):
            dR = e[tag]['R'] - e['lam0']['R']
            e[tag]['sign_rule'] = bool(np.sign(dR) == np.sign(e[tag]['delta'] * Kcl))
        dbal = abs(e['lam1_balanced']['R'] / e['lam0']['R'] - 1)
        e['balanced_vs_lam0'] = dbal
        say(f'    sign rule holds: equal {e["lam1_equal"]["sign_rule"]}, '
            f'T/gamma {e["lam1_tolman"]["sign_rule"]}; balanced vs lambda->0: {dbal:.1e}')
        out[f'{O1:.2f},{O2:.2f}'] = e
    return out


def stage_checks(say):
    out = {'shooting': [], 'tolerance': [], 'continuation': [], 'small_lambda': [],
           'zero_flux_tol': [], 'balanced': []}
    say('collocation vs two-parameter shooting (all table cases):')
    for O1, O2 in CASES:
        for lam, prot, tag in HEAT:
            h = solve_heat(R1, R2, O1, O2, lam, prot)
            p, err = shoot_heat(R1, R2, O1, O2, lam, prot, [h['KL'], h['KE']])
            d = max(abs(p[0] / h['KL'] - 1), abs(p[1] / h['KE'] - 1))
            out['shooting'].append(dict(O1=O1, O2=O2, tag=tag, KL_bvp=h['KL'], KL_shoot=p[0],
                                        KE_bvp=h['KE'], KE_shoot=p[1], rel=d))
            say(f'  ({O1},{O2}) {tag:14s} K_L {h["KL"]:.10f} / {p[0]:.10f}  '
                f'K_E {h["KE"]:.10f} / {p[1]:.10f}  rel diff {d:.1e}')
    say('collocation tolerance 1e-6 vs 1e-8 vs 1e-9:')
    for v, lam, prot in [(0.9, 1.0, 'equal'), (0.99, 0.0, 'equal'), (0.99, 1.0, 'tolman')]:
        Rs = [solve_heat(R1, R2, v, 0.0, lam, prot, tol=t)['R'] for t in (1e-6, 1e-8, 1e-9)]
        out['tolerance'].append(dict(v=v, lam=lam, prot=prot, R=Rs))
        say(f'  v1={v} lam={lam} {prot}: ' + '  '.join(f'{x:.10f}' for x in Rs))
    say('continuation step count (default vs doubled):')
    for v, lam, prot in [(0.99, 1.0, 'equal'), (0.999, 0.0, 'equal')]:
        a = solve_heat(R1, R2, v, 0.0, lam, prot)
        n = int(40 + 60 * np.arctanh(v))
        b = solve_heat(R1, R2, v, 0.0, lam, prot, nsteps=2 * n)
        out['continuation'].append(dict(v=v, lam=lam, prot=prot, R=[a['R'], b['R']]))
        say(f'  v1={v} lam={lam} {prot}: {a["R"]:.10f}  {b["R"]:.10f}')
    say('zero-heat-flux constant vs ODE tolerance (rtol 1e-10, 1e-12, 1e-13):')
    global RTOL, ATOL
    for v in (0.5, 0.9, 0.99):
        Ks = []
        for rt in (1e-10, 1e-12, 1e-13):
            RTOL, ATOL = rt, rt * 1e-2
            Ks.append(solve_zero_flux(R1, R2, v, 0.0)['K'])
        RTOL, ATOL = 1e-12, 1e-14
        out['zero_flux_tol'].append(dict(v=v, K=Ks))
        say(f'  v1={v}: ' + '  '.join(f'{k:.13f}' for k in Ks)
            + f'  rel(1e-12 vs 1e-13) {abs(Ks[1] / Ks[2] - 1):.1e}')
    say('balanced walls reproduce the lambda -> 0 solution:')
    for v, lams in ((0.5, (1.0, 5.0, 20.0)), (0.9, (1.0, 5.0, 20.0)), (0.99, (1.0, 5.0))):
        h0 = solve_heat(R1, R2, v, 0.0, 0.0, 'equal')
        for lam in lams:
            h = solve_heat(R1, R2, v, 0.0, lam, 'balanced')
            d = max(abs(h['KL'] / h0['KL'] - 1), abs(h['KE'] / h0['KE'] - 1))
            out['balanced'].append(dict(v=v, lam=lam, rel=d, T2=h['T2']))
            say(f'  v1={v} lambda={lam:g}: rel diff in (K_L, K_E) {d:.1e}; T(R2)/T(R1) = {h["T2"]:.4f}')
    say('approach to the lambda -> 0 limit (v1 = 0.9, equal walls):')
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        h = solve_heat(R1, R2, 0.9, 0.0, lam, 'equal')
        out['small_lambda'].append((lam, h['R']))
        say(f'  lambda={lam:g}: R={h["R"]:.8f}')
    say(f'  lambda=0 : R={solve_heat(R1, R2, 0.9, 0.0, 0.0, "equal")["R"]:.8f}')
    return out


def stage_curves(say):
    out = {}
    vgrid = np.tanh(np.linspace(np.arctanh(0.02), np.arctanh(0.999), 80))
    out['zero_flux'] = [(v, solve_zero_flux(R1, R2, v, 0.0)['R']) for v in vgrid]
    runs = [(0.0, 'equal', True, 'lam0', 0.999), (1.0, 'equal', True, 'lam1_equal', 0.999),
            (1.0, 'tolman', True, 'lam1_tolman', 0.995),
            (0.0, 'equal', False, 'lam0_noinertia', 0.999)]
    for lam, prot, inert, tag, vmax in runs:
        h = solve_heat(R1, R2, vmax, 0.0, lam, prot, inertial=inert, record=True,
                       nsteps=int(60 + 80 * np.arctanh(vmax)))
        out[tag] = [(o1, KL / classical(R1, R2, o1, 0.0)[2], KE) for o1, o2, KL, KE in h['path']]
        say(f'  {tag:15s} R(v1={vmax}) = {h["R"]:.4f}')
    say('  zero flux at 0.5/0.9/0.99/0.995/0.999: ' + ', '.join(
        f'{solve_zero_flux(R1, R2, v, 0.0)["R"]:.4f}' for v in (0.5, 0.9, 0.99, 0.995, 0.999)))
    out['inertia_effect'] = []
    for v in (0.3, 0.5, 0.9, 0.99, 0.999):
        a = solve_heat(R1, R2, v, 0.0, 0.0, 'equal')['R']
        b = solve_heat(R1, R2, v, 0.0, 0.0, 'equal', inertial=False)['R']
        out['inertia_effect'].append((v, a, b))
        say(f'  lambda->0, v1={v}: with inertial term {a:.6f}, without {b:.6f} (ratio {a / b:.4f})')
    out['lambda_scan'] = {}
    lams = [0.0, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0]
    for prot in ('equal', 'tolman', 'balanced'):
        out['lambda_scan'][prot] = [(l, solve_heat(R1, R2, 0.9, 0.0, l, prot)['R']) for l in lams]
        say(f'  v1=0.9 {prot:8s}: ' + ', '.join(f'lam={l:g}: {x:.4f}'
                                                  for l, x in out['lambda_scan'][prot]))
    return out


def stage_thermal(say):
    out = {'inv_mu2': [], 'dT_scan': [], 'balanced_dT': []}
    for v in (0.5, 0.9, 0.99):
        h = solve_heat(R1, R2, v, 0.0, 0.0, 'equal')
        out['inv_mu2'].append((v, 1 / h['mu2'], 1 / np.sqrt(1 - v * v)))
        say(f'  v1={v}: 1/mu(R2) = {1 / h["mu2"]:.4f} (gamma1 = {1 / np.sqrt(1 - v * v):.4f}), '
            f'max Theta(lambda->0) = {h["Theta"].max():.4f}')
    for v in (0.5, 0.9, 0.99):
        for lam in (1.0, 3.0, 10.0, 30.0):
            h = solve_heat(R1, R2, v, 0.0, lam, 'equal')
            try:
                b = solve_heat(R1, R2, v, 0.0, lam, 'balanced')
                bdT, bT2 = b['dT_max'], b['T2']
            except RuntimeError:                 # very large Theta: mesh limit reached
                bdT, bT2 = None, None
            out['dT_scan'].append(dict(v=v, lam=lam, Br=v * v / lam, dT_equal=h['dT_max'],
                                       R_equal=h['R'], dT_balanced=bdT, T2_balanced=bT2))
            say(f'  v1={v} lambda={lam:g} Br={v * v / lam:.3f}: equal walls dT_max/T_w={h["dT_max"]:.3f} '
                f'(R={h["R"]:.3f}); balanced walls T2/T1={bT2}, rise above T2 {bdT}')
    return out


def stage_profiles(say):
    out = {}
    for v in (0.5, 0.9, 0.99):
        zf = solve_zero_flux(R1, R2, v, 0.0, npts=401)
        h0 = solve_heat(R1, R2, v, 0.0, 0.0, 'equal')
        h1 = solve_heat(R1, R2, v, 0.0, 1.0, 'equal')
        h1t = solve_heat(R1, R2, v, 0.0, 1.0, 'tolman')
        h1b = solve_heat(R1, R2, v, 0.0, 1.0, 'balanced')
        A, B, _ = classical(R1, R2, v, 0.0)
        r = zf['r']
        sl = slice(None, None, 10)
        out[str(v)] = dict(
            r=r.tolist(), classical=(r * (A + B / r**2)).tolist(),
            zero_flux=(r * zf['Omega']).tolist(), rh=h0['r'][sl].tolist(),
            lam0=(h0['r'] * h0['Omega'])[sl].tolist(),
            lam1=(h1['r'] * h1['Omega'])[sl].tolist(),
            theta_lam0=h0['Theta'][sl].tolist(), T_lam1=h1['T'][sl].tolist(),
            T_lam1_tolman=h1t['T'][sl].tolist(), T_lam1_balanced=h1b['T'][sl].tolist())
        th = h0['Theta']
        say(f'  v1={v}: max Theta(lambda->0) = {th.max():.4f} at r={h0["r"][np.argmax(th)]:.3f}; '
            f'lambda=1 dT_max/T_w: equal {h1["dT_max"]:.4f}, T/gamma {h1t["dT_max"]:.4f}, '
            f'balanced {h1b["dT_max"]:.4f}')
    return out


def stage_pn(say):
    pc = pn_corrections(R1, R2, 1.0, 0.0, 1.0)         # coefficients of v1^2
    p05 = pn_corrections(R1, R2, 0.5, 0.0, 1.0)
    say(f'  integrals (Omega1 = 0.5): I1={p05["I1"]:.5f} I2={p05["I2"]:.5f} '
        f'I3={p05["I3"]:.5f} I_T={p05["IT"]:.5f} l={p05["ell"]:.5f}')
    say(f'  R-1 at Omega1=0.5: zero flux {p05["zero_flux"]:.5f}, lambda->0 {p05["heat_lam0"]:.5f}, '
        f'equal walls +{p05["heat_equal"] - p05["heat_lam0"]:.5f} lambda, '
        f'equal T/gamma +{p05["heat_tolman"] - p05["heat_lam0"]:.5f} lambda')
    vs = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3])
    out = {'v': vs.tolist(), 'analytic': {}, 'at_0.5': p05}
    keys = ('zero_flux', 'heat_lam0', 'heat_equal', 'heat_tolman')
    out['zero_flux'] = [(solve_zero_flux(R1, R2, v, 0.0)['R'] - 1) / v**2 for v in vs]
    for lam, prot, k in [(0.0, 'equal', 'heat_lam0'), (1.0, 'equal', 'heat_equal'),
                         (1.0, 'tolman', 'heat_tolman')]:
        out[k] = [(solve_heat(R1, R2, v, 0.0, lam, prot, nsteps=20, tol=1e-10)['R'] - 1) / v**2
                  for v in vs]
    for k in keys:
        c = float(np.polyfit(vs[:7]**2, out[k][:7], 2)[-1])
        out['analytic'][k] = pc[k]
        out[k + '_limit'] = c
        say(f'  {k:12s}: extrapolated (R-1)/v1^2 -> {c:.6f}, analytic {pc[k]:.6f}, '
            f'rel diff {abs(c / pc[k] - 1):.1e}')
    return out


def stage_planar(say):
    v1 = 0.9
    out = {'v1': v1, 'eps': [], 'err_heat': [], 'err_zero': [], 'profiles': {}}
    for eps in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
        R2n = R1 * (1 + eps)
        h = solve_heat(R1, R2n, v1 / R1, 0.0, 1.0, 'equal')
        zf = solve_zero_flux(R1, R2n, v1 / R1, 0.0, npts=2001)
        eh = float(np.max(np.abs(h['r'] * h['Omega'] - planar_heat_profile(h['r'] - R1, eps, v1))))
        ez = float(np.max(np.abs(zf['r'] * zf['Omega'] -
                                 planar_zero_flux_profile(zf['r'] - R1, eps, v1))))
        out['eps'].append(eps); out['err_heat'].append(eh); out['err_zero'].append(ez)
        out['profiles'][str(eps)] = dict(x=((h['r'] - R1) / eps)[::40].tolist(),
                                         v_heat=(h['r'] * h['Omega'])[::40].tolist())
        say(f'  d/R1={eps:<7g} max|v - v_planar|: heat {eh:.3e}, zero flux {ez:.3e}')
    e = np.array(out['eps'])
    for k in ('err_heat', 'err_zero'):
        slope = np.polyfit(np.log(e[-3:]), np.log(np.array(out[k])[-3:]), 1)[0]
        out[k + '_slope'] = float(slope)
        say(f'  {k}: log-log slope over the three narrowest gaps = {slope:.3f}')
    x = np.linspace(0, 1, 201)
    out['x'] = x.tolist()
    out['planar_heat'] = planar_heat_profile(x, 1.0, v1).tolist()
    out['planar_zero'] = planar_zero_flux_profile(x, 1.0, v1).tolist()
    return out


def stage_extras(say):
    """Diagnostics quoted in the discussion of the results."""
    from scipy.optimize import brentq as _brentq
    out = {}
    # sign of the heat flux at the inner wall: K_E/K_L against Omega1 at 0.9c
    out['KE_over_KL'] = {}
    for prot in ('equal', 'tolman'):
        ratio = lambda l, p=prot: (lambda h: h['KE'] / h['KL'])(solve_heat(R1, R2, 0.9, 0.0, l, p))
        vals = [(l, ratio(l)) for l in (0.0, 1.0, 10.0)]
        lam_star = _brentq(lambda l: ratio(l) - 0.9, 1.0, 10.0, xtol=1e-4)
        out['KE_over_KL'][prot] = dict(values=vals, lambda_reversal=lam_star)
        say(f'  v1=0.9 {prot:6s}: K_E/K_L = ' + ', '.join(f'{x:.4f} (lam={l:g})' for l, x in vals)
            + f'; q(R1) changes sign at lambda = {lam_star:.2f}')
    h0 = solve_heat(R1, R2, 0.9, 0.0, 0.0, 'equal')
    out['q_walls_lam0'] = (h0['q'][0], h0['q'][-1])
    say(f'  v1=0.9 lambda->0: q(R1) = {h0["q"][0]:.4f}, q(R2) = {h0["q"][-1]:.4f} (units eta c^2/R1)')
    # case (0, 0.2), lambda = 1, equal walls: temperature below T_w
    h = solve_heat(R1, R2, 0.0, 0.2, 1.0, 'equal')
    T, r = h['T'], h['r']
    out['case_0_02'] = dict(T_min=float(T.min()), r_min=float(r[T.argmin()]),
                            rise=float(T.max() - 1), frac_below=float(np.mean(T < 1)))
    say(f'  (0,0.2) lambda=1 equal: T_min/T_w = {T.min():.4f} at r = {r[T.argmin()]:.3f}; '
        f'T < T_w over {100 * np.mean(T < 1):.0f}% of the gap; max rise {T.max() - 1:.2e}')
    # local slopes of the torque ratio against gamma1 (log-log)
    C = json.load(open(os.path.join('results', 'curves.json')))
    G = lambda v: 1 / np.sqrt(1 - v * v)

    def slope(curve, v1, v2):
        d = np.array(curve)
        lr = np.interp(np.log([G(v1), G(v2)]), np.log(G(d[:, 0])), np.log(d[:, 1]))
        return float((lr[1] - lr[0]) / np.log(G(v2) / G(v1)))
    out['slopes'] = {k: (slope(C[k], 0.99, 0.999), slope(C[k], 0.995, 0.999))
                     for k in ('zero_flux', 'lam0', 'lam1_equal', 'lam0_noinertia')}
    for k, (a, b) in out['slopes'].items():
        say(f'  d ln R / d ln gamma1, {k:15s}: {a:.3f} (0.99c-0.999c), {b:.3f} (0.995c-0.999c)')
    # factors multiplying chi = eta c/[(e+P) R1] in the inverse Reynolds numbers
    out['inverse_reynolds'] = []
    for v in (0.5, 0.9, 0.99):
        row = dict(v=v)
        for lam, key in ((0.0, 'lam0'), (1.0, 'lam1_equal')):
            h = solve_heat(R1, R2, v, 0.0, lam, 'equal')
            r, Om = h['r'], h['Omega']
            g = 1 / np.sqrt(1 - (r * Om)**2)
            row[key] = (float(np.max(r * g**2 * np.abs(np.gradient(Om, r, edge_order=2)))),
                        float(np.max(np.abs(h['q']))))
        out['inverse_reynolds'].append(row)
        say(f'  v1={v}: max r gamma^2|Omega\'| and max|q| = {row["lam0"][0]:.2f}, {row["lam0"][1]:.2f} '
            f'(lambda->0); {row["lam1_equal"][0]:.2f}, {row["lam1_equal"][1]:.2f} (lambda=1, equal T)')
    return out


STAGES = dict(cases=stage_cases, checks=stage_checks, curves=stage_curves,
              thermal=stage_thermal, profiles=stage_profiles, pn=stage_pn, planar=stage_planar,
              extras=stage_extras)


def main(argv):
    force = '--force' in argv
    names = [a for a in argv if not a.startswith('--')] or list(STAGES)
    os.makedirs('results', exist_ok=True)
    log = open(os.path.join('results', 'summary.txt'), 'a')

    def say(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        log.write(line + '\n')

    for name in names:
        path = os.path.join('results', f'{name}.json')
        if os.path.exists(path) and not force:
            print(f'[{name}] cached in {path}; use --force to recompute')
            continue
        t0 = time.time()
        say(f'=== {name} ({time.strftime("%Y-%m-%d %H:%M")}) ===')
        data = STAGES[name](say)
        with open(path, 'w') as fh:
            json.dump(data, fh, default=float, indent=1)
        say(f'[{name}] {time.time() - t0:.0f} s')
    log.close()


if __name__ == '__main__':
    main(sys.argv[1:])
