"""
make_figures.py -- figures and LaTeX tables of the paper from results/*.json.

Run after rtc_solver.py:   python make_figures.py
Writes figures/fig_profiles.pdf, fig_torque.pdf, fig_temperature.pdf and
fig_limits.pdf (Figs. 1-4 of the paper, with .png previews) and results/tables.tex.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colour order fixed across all figures (checked for colour-vision separation);
# every series also has its own line style, so the figures read in grey scale.
C = dict(classical='#6b6b6b', zero='#2a78d6', lam0='#eb6834', eq='#1baf7a',
         tol='#eda100', ink='#222222')
LS = dict(classical=(0, (4, 2)), zero='-', lam0='-', eq=(0, (6, 1.5)),
          tol=(0, (1.5, 1.2)), noinert=(0, (3, 1.5, 1, 1.5)))
COL_W, PAGE_W = 3.375, 7.0                       # PRD column and page widths (in)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif',
                                           'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 8.5, 'axes.labelsize': 9,
    'legend.fontsize': 7.2, 'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.3, 'legend.frameon': False, 'savefig.dpi': 300,
    'pdf.fonttype': 42,
})


def load(name):
    with open(os.path.join('results', f'{name}.json')) as fh:
        return json.load(fh)


def save(fig, name):
    os.makedirs('figures', exist_ok=True)
    fig.savefig(os.path.join('figures', name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join('figures', name + '.png'), bbox_inches='tight', pad_inches=0.02,
                dpi=200)
    plt.close(fig)


def panel_label(ax, s, x=0.03, y=0.95):
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va='top', fontsize=9)


# ----------------------------------------------------------------------------
def fig_profiles():
    P = load('profiles')
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.25), sharey=True)
    for ax, (v, lab) in zip(axes, [('0.5', '(a)'), ('0.9', '(b)'), ('0.99', '(c)')]):
        d = P[v]
        v1 = float(v)
        r, rh = np.array(d['r']), np.array(d['rh'])
        ax.plot(r, np.array(d['classical']) / v1, color=C['classical'], ls=LS['classical'],
                lw=1.1, label='classical')
        ax.plot(r, np.array(d['zero_flux']) / v1, color=C['zero'], ls=LS['zero'],
                label='zero heat flux')
        ax.plot(rh, np.array(d['lam0']) / v1, color=C['lam0'], ls=LS['lam0'],
                label=r'heat, $\lambda\to0$')
        ax.plot(rh, np.array(d['lam1']) / v1, color=C['eq'], ls=LS['eq'],
                label=r'heat, $\lambda=1$, equal $T$')
        ax.set_xlim(1, 2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r'$r/R_1$')
        ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8] + ([2.0] if lab == '(c)' else []))
        ax.set_title(f'{lab} $v_1={v}c$', fontsize=9, pad=4)
    axes[0].set_ylabel(r'$v_\phi/v_1$')
    axes[0].legend(loc='lower left', handlelength=2.6)
    fig.subplots_adjust(wspace=0.06)
    save(fig, 'fig_profiles')


def fig_torque():
    Cv = load('curves')
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 4.6))
    g = lambda v: 1 / np.sqrt(1 - np.asarray(v)**2)
    z = np.array(Cv['zero_flux'])
    a.loglog(g(z[:, 0]), z[:, 1], color=C['zero'], ls=LS['zero'], label='zero heat flux')
    for key, col, ls, lab in [('lam0', 'lam0', LS['lam0'], r'heat, $\lambda\to0$'),
                              ('lam1_equal', 'eq', LS['eq'], r'heat, $\lambda=1$, equal $T$'),
                              ('lam1_tolman', 'tol', LS['tol'], r'heat, $\lambda=1$, equal $T/\gamma$'),
                              ('lam0_noinertia', 'lam0', LS['noinert'],
                               r'heat, $\lambda\to0$, $\mu=1$')]:
        d = np.array(Cv[key])
        a.loglog(g(d[:, 0]), d[:, 1], color=C[col], ls=ls, label=lab,
                 lw=1.0 if key == 'lam0_noinertia' else 1.3)
    a.set_xlim(1, 25)
    a.set_ylim(0.9, 4e4)
    a.set_xlabel(r'$\gamma_1=(1-v_1^2/c^2)^{-1/2}$')
    a.set_ylabel(r'$\mathcal{R}=K_L/K_{\mathrm{cl}}$')
    a.legend(loc='upper left', handlelength=2.6, bbox_to_anchor=(0.09, 1.0))
    panel_label(a, '(a)')
    # (b) dependence on lambda at v1 = 0.9c
    for prot, col, ls, lab in [('tolman', 'tol', LS['tol'], r'equal $T/\gamma$'),
                               ('equal', 'eq', LS['eq'], r'equal $T$'),
                               ('balanced', 'lam0', LS['lam0'], 'balanced walls')]:
        d = np.array(Cv['lambda_scan'][prot])
        b.plot(d[:, 0], d[:, 1], color=C[col], ls=ls, marker='o', ms=3.2, label=lab)
    zf09 = load('cases')['0.90,0.00']['zero_flux']['R']
    b.axhline(zf09, color=C['zero'], ls=LS['zero'], lw=1.0)
    b.text(9.8, zf09 + 0.5, 'zero heat flux', color=C['ink'], ha='right', va='bottom',
           fontsize=7.2)
    b.set_xlim(-0.2, 10.4)
    b.set_ylim(0, 28)
    b.set_xlabel(r'$\lambda=\kappa T_w/(\eta c^2)$')
    b.set_ylabel(r'$\mathcal{R}_q$ at $v_1=0.9c$')
    b.legend(loc='upper left', handlelength=2.6, bbox_to_anchor=(0.09, 1.0))
    panel_label(b, '(b)')
    fig.subplots_adjust(hspace=0.32)
    save(fig, 'fig_torque')


def fig_temperature():
    P = load('profiles')
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 4.3))
    cols = {'0.5': C['zero'], '0.9': C['lam0'], '0.99': C['eq']}
    lss = {'0.5': '-', '0.9': (0, (6, 1.5)), '0.99': (0, (1.5, 1.2))}
    for v in ('0.5', '0.9', '0.99'):
        d = P[v]
        th = np.array(d['theta_lam0'])
        a.plot(d['rh'], th / th.max(), color=cols[v], ls=lss[v],
               label=rf'$v_1={v}c$, $\Theta_{{\max}}={th.max():#.3g}$')
    a.set_xlim(1, 2)
    a.set_ylim(0, 1.12)
    a.set_xlabel(r'$r/R_1$')
    a.set_ylabel(r'$\Theta/\Theta_{\max}$')
    a.legend(loc='lower center', handlelength=2.6)
    panel_label(a, r'(a) $\lambda\to0$')
    d = P['0.9']
    for key, col, ls, lab in [('T_lam1_tolman', C['tol'], LS['tol'], r'equal $T/\gamma$'),
                              ('T_lam1', C['eq'], LS['eq'], r'equal $T$'),
                              ('T_lam1_balanced', C['lam0'], LS['lam0'], 'balanced walls')]:
        b.plot(d['rh'], d[key], color=col, ls=ls, label=lab)
    b.set_xlim(1, 2)
    b.set_ylim(0.8, 4.5)
    b.set_xlabel(r'$r/R_1$')
    b.set_ylabel(r'$T/T_w$')
    b.legend(loc='upper right', handlelength=2.6)
    panel_label(b, r'(b) $\lambda=1$, $v_1=0.9c$')
    fig.subplots_adjust(hspace=0.32)
    save(fig, 'fig_temperature')


def fig_limits():
    """Fig. 4 of the paper: weakly relativistic and narrow-gap limits in one row."""
    D, Pl = load('pn'), load('planar')
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(PAGE_W, 2.35))
    v = np.array(D['v'])
    for k, col, ls, lab in [('heat_tolman', 'tol', LS['tol'], r'heat, $\lambda=1$, equal $T/\gamma$'),
                            ('zero_flux', 'zero', '-', 'zero heat flux'),
                            ('heat_equal', 'eq', LS['eq'], r'heat, $\lambda=1$, equal $T$'),
                            ('heat_lam0', 'lam0', '-', r'heat, $\lambda\to0$')]:
        a.plot(v, D[k], color=C[col], ls=ls, marker='o', ms=2.4, label=lab)
        a.plot([0, 0.06], [D['analytic'][k]] * 2, color=C['ink'], lw=0.8, ls=':')
    a.set_xlim(0, 0.31)
    a.set_ylim(0.45, 1.08)
    a.set_xlabel(r'$v_1/c$')
    a.set_ylabel(r'$(\mathcal{R}-1)\,c^2/v_1^2$')
    a.legend(loc='upper left', handlelength=2.2, fontsize=6.4, labelspacing=0.25)
    a.set_title('(a)', fontsize=9, pad=3)
    x = np.array(Pl['x'])
    b.plot(x, np.array(Pl['planar_heat']) / Pl['v1'], color=C['ink'], lw=1.0, label='planar Eckart', zorder=5)
    for eps, col, ls in [('0.4', C['zero'], (0, (6, 1.5))), ('0.1', C['lam0'], (0, (3, 1.5, 1, 1.5))),
                         ('0.025', C['eq'], (0, (1.5, 1.2)))]:
        p = Pl['profiles'][eps]
        b.plot(p['x'], np.array(p['v_heat']) / Pl['v1'], color=col, ls=ls, label=rf'$d/R_1={eps}$')
    b.set_xlim(0, 1)
    b.set_ylim(0, 1.02)
    b.set_xlabel(r'$(r-R_1)/d$')
    b.set_ylabel(r'$v_\phi/v_1$')
    b.legend(loc='lower left', handlelength=2.4, fontsize=6.6)
    b.set_title(r'(b) heat, $\lambda=1$, equal $T$', fontsize=9, pad=3)
    e = np.array(Pl['eps'])
    c.loglog(e, Pl['err_zero'], color=C['zero'], marker='s', ms=3.0, ls='-', label='zero heat flux')
    c.loglog(e, Pl['err_heat'], color=C['lam0'], marker='o', ms=3.0, ls='-', label='heat')
    ref = Pl['err_heat'][-1] * e / e[-1] * 0.55
    c.loglog(e, ref, color=C['ink'], lw=0.8, ls=':')
    c.text(0.06, ref[3] * 0.55, r'$\propto d/R_1$', fontsize=7.2, ha='left', va='top')
    c.set_xticks(e)
    c.set_xticklabels([f'{x:g}' for x in e], fontsize=6.8)
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xlabel(r'$d/R_1$')
    c.set_ylabel(r'$\max_r|v_\phi-v_{\mathrm{planar}}|/c$')
    c.legend(loc='upper left', handlelength=2.0, fontsize=6.8)
    c.set_title('(c)', fontsize=9, pad=3)
    fig.subplots_adjust(wspace=0.5)
    save(fig, 'fig_limits')


# ----------------------------------------------------------------------------
def fmt(x):
    """Three significant figures; powers of ten below 0.01."""
    if x is None:
        return '---'
    if abs(x) < 0.01:
        m, e = f'{x:.1e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'
    return f'${x:.3g}$' if abs(x) < 1 else f'${x:.2f}$'


def tables():
    Cs = load('cases')
    lab = lambda e: f"$({e['O1']:.2f},\\,{e['O2']:.2f})$"
    lines = ['% generated by make_figures.py from results/cases.json',
             '% --- Table I: zero-heat-flux model ---']
    for e in Cs.values():
        z = e['zero_flux']
        lines.append(f"{lab(e)} & {'yes' if e['rayleigh_stable'] else 'no'} & "
                     f"{fmt(z['dOmega'])} & ${z['R']:.4f}$ \\\\")
    lines.append('% --- Table II: heat-conducting model ---')
    for e in Cs.values():
        a, b, c = e['lam0'], e['lam1_equal'], e['lam1_tolman']
        lines.append(f"{lab(e)} & ${a['R']:.4f}$ & ${b['R']:.4f}$ & ${c['R']:.4f}$ & "
                     f"${e['Br_lam1']:.2f}$ & {fmt(b['dT_max'])} \\\\")
    lines.append('% --- Table III: benchmark values ---')
    names = [('lam0', r'$\lambda\to0$'), ('lam1_equal', r'$\lambda=1$, equal $T$'),
             ('lam1_tolman', r'$\lambda=1$, equal $T/\gamma$')]
    for e in Cs.values():
        lines.append(f"{lab(e)} & zero heat flux & ${e['zero_flux']['K']:.10f}$ & --- \\\\")
        for k, n in names:
            lines.append(f" & {n} & ${e[k]['KL']:.10f}$ & ${e[k]['KE']:.10f}$ \\\\")
        lines.append('\\hline')
    with open(os.path.join('results', 'tables.tex'), 'w') as fh:
        fh.write('\n'.join(lines[:-1]) + '\n')
    print('\n'.join(lines[:-1]))


if __name__ == '__main__':
    fig_profiles()
    fig_torque()
    fig_temperature()
    fig_limits()
    tables()
    print('figures written to figures/')
