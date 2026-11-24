# Special-relativistic Taylor–Couette flow

Numerical solver and figure scripts for

> A. Gelekva, *Special-relativistic Taylor–Couette flow: Viscous shear, radial
> heat conduction, and energy conservation* (2026).

The code solves the stationary laminar flow of a viscous relativistic fluid
between two coaxial rotating cylinders in two formulations:

* the **zero-heat-flux model**, `eta r^3 gamma^3 Omega' = -K_L0`;
* the **heat-conducting model** in the Eckart frame, with the two first
  integrals `r T^{rt} = K_E` and `r^3 T^{r phi} = K_L`, the reduced equation
  `eta r^3 gamma Omega' = r^2 Omega K_E - K_L`, and the temperature equation
  `kappa (T' - r gamma^2 Omega^2 T) = -(gamma/r)(K_E - Omega K_L)`.

It reproduces every number, table and figure of the paper.

## Requirements

Python 3.10 or later, NumPy, SciPy (1.12 or later), Matplotlib:

```
pip install -r requirements.txt
```

## Usage

```
python rtc_solver.py          # all stages (about 10 min); cached stages are skipped
python rtc_solver.py --force  # recompute everything
python rtc_solver.py pn       # one stage only
python make_figures.py        # Figs. 1-4 and the LaTeX table rows
```

Units: `c = eta = 1`, `R1 = 1`, wall temperature scale `T_w = 1`, so that the
dimensionless conductivity `lambda = kappa T_w/(eta c^2)` equals `kappa`.
The results of a full run are included in `results/` and `figures/`.

## Thermal protocols

| key | wall temperatures | note |
|---|---|---|
| `equal` | `T(R1) = T(R2) = T_w` | |
| `tolman` | `T(R_i) = gamma_i T_w` | "equal T/gamma" in the paper |
| `balanced` | `T(R2)/T(R1) = 1/mu(R2)` | imposed self-consistently; gives the lambda -> 0 solution for every lambda |
| `lambda = 0` | `Theta = kappa T/(eta c^2) = 0` at both walls | limit kappa -> 0 at fixed wall temperatures |

## Stages

| Stage | Content | Paper |
|---|---|---|
| `cases` | four cases (0.5,0.2), (0,0.2), (0.5,0), (0.9,0); both models, all thermal settings and accuracy checks | Tables I, II, IV |
| `checks` | shooting vs. collocation, tolerance and continuation tests, balanced walls vs. lambda -> 0 | Table III |
| `curves` | torque ratio vs. wall speed up to 0.999c; dependence on lambda at 0.9c | Fig. 2 |
| `thermal` | 1/mu(R2) and temperature rise vs. lambda | Sec. IV C |
| `profiles` | velocity and temperature profiles at 0.5c, 0.9c, 0.99c | Figs. 1, 3 |
| `pn` | weakly relativistic limit against the analytic O(v^2/c^2) coefficients | Fig. 4(a) |
| `planar` | narrow-gap limit against the planar closed-form solutions | Fig. 4(b,c) |
| `extras` | sign of the heat flux at the inner wall, temperature dip for (0,0.2), log-log slopes, inverse-Reynolds factors | Secs. V D, VI B |

Each stage writes `results/<stage>.json` and appends a log to
`results/summary.txt`.

## Methods

* **Zero-heat-flux model:** shooting inwards from the outer wall (DOP853,
  relative tolerance 1e-12) with Brent's method on `K_L0`.
* **Heat-conducting model:** collocation with `scipy.integrate.solve_bvp`
  (tolerance 1e-9) for the rapidity `y = artanh(r Omega)`, `Theta = kappa T/(eta c^2)`
  and `M = int r gamma^2 Omega^2 dr` (so that `mu = exp(-M)`), with
  `(K_L, K_E)` as unknown parameters and continuation in the wall rapidities;
  cross-checked by two-parameter shooting (`shoot_heat`).

## Known limits

* With the default settings the heat-conducting model converges to 0.999c for
  lambda -> 0 and for lambda = 1 with equal wall temperatures, and to 0.995c for
  lambda = 1 with equal T/gamma.
* Too large continuation steps can land on spurious solutions with very large
  `K` and small residuals. The default step counts were tested by doubling
  them; repeat that test if the geometry is changed.

## License

MIT; see `LICENSE`.
