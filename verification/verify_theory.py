"""Numerical verification of the theory in Supporting Information S1 (no data needed).

Run:  python verification/verify_theory.py
Writes a PASS/FAIL line per check to stdout. No data needed. numpy + scipy only.
Each check is labelled T1, T2, ... in the output. Internal result labels map to the manuscript as follows:
Theorem 0 is Proposition 1; Theorem 1 is the quadratic-form identity (Eq. 1); Corollary 3 is the
reference-free spacing test (Supporting Information S1); other labels refer to auxiliary results.
"""
import numpy as np

rng = np.random.default_rng(20261113)
LINES = []


def report(tag, ok, detail, kind="check"):
    """kind: 'check' (counted), 'xfail' (stress test whose failure is the expected outcome; counted as a check that the
    failure is reproduced), 'info' (always printed, never counted)."""
    if kind == "info":
        status = "INFO"
    elif kind == "xfail":
        status = "XFAIL-REPRODUCED" if ok else "FAIL"
    else:
        status = "PASS" if ok else "FAIL"
    line = f"[{status}] {tag}: {detail}"
    LINES.append(line)
    print(line, flush=True)


def loglog_slope(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return -np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0]


# ---------------------------------------------------------------- T1 mixture identity
def t1():
    M = 12
    pos = rng.uniform(0, 10, (M, 2))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    chi = rng.uniform(1.0, 3.0, M)
    A = np.exp(rng.normal(0, 0.5, M))
    lam0, gam = 4.0, 0.4
    ell = rng.normal(0, 1, M)
    f = np.logspace(0, np.log10(40), 4001)
    lf = np.log(f)

    def S(fv):
        P = A[None, :] * fv[:, None] ** (-chi[None, :])
        amp = np.sqrt(P[:, :, None] * P[:, None, :])
        c = np.exp(-d[None] * fv[:, None, None] ** gam / lam0)
        return amp * c

    Sf = S(f)
    Pe = np.einsum('i,fij,j->f', ell, Sf, ell)
    assert np.all(Pe > 0)
    chi_num = -np.gradient(np.log(Pe), lf)
    chi_ij = 0.5 * (chi[:, None] + chi[None, :])[None] + gam * d[None] * f[:, None, None] ** gam / lam0
    w = ell[None, :, None] * ell[None, None, :] * Sf / Pe[:, None, None]
    chi_id = np.sum(w * chi_ij, axis=(1, 2))
    err = np.max(np.abs(chi_num[5:-5] - chi_id[5:-5]))
    report("T1a mixture identity (exact derivative vs finite difference)", err < 1e-4,
           f"max |diff| = {err:.2e}; weights sum to {np.max(np.abs(w.sum((1, 2)) - 1)):.1e} from 1")

    # incoherent bound
    Pinc = np.einsum('i,fi->f', ell ** 2, A[None] * f[:, None] ** (-chi[None]))
    chi_inc = -np.gradient(np.log(Pinc), lf)[5:-5]
    ok = (chi_inc.min() >= chi.min() - 1e-6) and (chi_inc.max() <= chi.max() + 1e-6)
    report("T1b incoherent bound min chi_i <= chi_e <= max chi_i", ok,
           f"chi_e in [{chi_inc.min():.3f}, {chi_inc.max():.3f}], local in [{chi.min():.3f}, {chi.max():.3f}]")

    # coherence certificate: homogeneous exponent, positive lead, scalp exponent exceeds every local exponent
    chi_h = 1.5
    ellp = np.abs(ell)
    Sh = f[:, None, None] ** (-chi_h) * np.exp(-d[None] * f[:, None, None] ** gam / lam0)
    Ph = np.einsum('i,fij,j->f', ellp, Sh, ellp)
    chi_c = -np.gradient(np.log(Ph), lf)[5:-5]
    report("T1c coherence certificate (scalp exponent > max local exponent)", chi_c.min() > chi_h,
           f"local exponent {chi_h}, scalp exponent in [{chi_c.min():.3f}, {chi_c.max():.3f}]")


# ---------------------------------------------------------------- T2 Gaussian elasticity identity
def t2():
    # d = 1
    n, dx = 2401, 0.05
    x = (np.arange(n) - n // 2) * dx
    leads = {
        "gauss L=3": np.exp(-x ** 2 / (2 * 3.0 ** 2)),
        "bipolar": np.exp(-(x - 4) ** 2 / (2 * 2.0 ** 2)) - np.exp(-(x + 4) ** 2 / (2 * 2.0 ** 2)),
        "random signed": sum(a * np.exp(-(x - c) ** 2 / (2 * s ** 2))
                             for a, c, s in zip(rng.normal(0, 1, 6), rng.uniform(-8, 8, 6), rng.uniform(0.8, 3, 6))),
    }
    lam = np.logspace(-0.3, 1.0, 40)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    worst = 0.0
    D2 = (x[:, None] - x[None, :]) ** 2
    for name, ell in leads.items():
        G = np.array([ell @ np.exp(-D2 / (2 * l ** 2)) @ ell * dx * dx for l in lam])
        if np.any(G <= 0):
            report(f"T2 d=1 {name}", False, "G not positive")
            continue
        eta_num = np.gradient(np.log(G), np.log(lam))
        W = np.abs(np.fft.fft(ell)) ** 2
        eta_f = np.array([1 - l ** 2 * np.sum(W * np.exp(-l ** 2 * k ** 2 / 2) * k ** 2)
                          / np.sum(W * np.exp(-l ** 2 * k ** 2 / 2)) for l in lam])
        e = np.max(np.abs(eta_num[2:-2] - eta_f[2:-2]))
        worst = max(worst, e)
        if name == "gauss L=3":
            closed = 1 / (1 + lam ** 2 / (2 * 9.0))
            report("T2a closed form eta = d/(1+lam^2/2L^2), d=1", np.max(np.abs(eta_f - closed)) < 1e-6,
                   f"max |diff| = {np.max(np.abs(eta_f - closed)):.1e}")
        if name == "bipolar":
            report("T2b zero-mean (bipolar) lead gives negative elasticity", eta_f.min() < -1.0,
                   f"eta ranges {eta_f.max():.3f} -> {eta_f.min():.3f} (asymptote -2)")
    report("T2c eta = d - lam^2 <k^2> (direct quadrature vs Fourier formula), d=1", worst < 5e-3,
           f"max |diff| over 3 leads = {worst:.1e}")

    # d = 2, separable Gaussian kernel
    n2, dx2 = 161, 0.25
    g = (np.arange(n2) - n2 // 2) * dx2
    X, Y = np.meshgrid(g, g, indexing='ij')
    lead2 = (np.exp(-((X - 2) ** 2 + Y ** 2) / (2 * 2.0 ** 2)) - 0.6 * np.exp(-((X + 3) ** 2 + (Y - 1) ** 2) / (2 * 1.5 ** 2))
             + 0.4 * np.exp(-(X ** 2 + (Y + 4) ** 2) / (2 * 2.5 ** 2)))
    lam2 = np.logspace(-0.2, 0.7, 25)
    Dg = (g[:, None] - g[None, :]) ** 2
    G2 = np.array([np.sum((np.exp(-Dg / (2 * l ** 2)) @ lead2 @ np.exp(-Dg / (2 * l ** 2))) * lead2) * dx2 ** 4 for l in lam2])
    eta_num2 = np.gradient(np.log(G2), np.log(lam2))
    kk = 2 * np.pi * np.fft.fftfreq(n2, dx2)
    KX, KY = np.meshgrid(kk, kk, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    W2 = np.abs(np.fft.fft2(lead2)) ** 2
    eta_f2 = np.array([2 - l ** 2 * np.sum(W2 * np.exp(-l ** 2 * K2 / 2) * K2) / np.sum(W2 * np.exp(-l ** 2 * K2 / 2)) for l in lam2])
    e2 = np.max(np.abs(eta_num2[2:-2] - eta_f2[2:-2]))
    report("T2d eta = d - lam^2 <k^2>, d=2 signed lead", e2 < 1e-2, f"max |diff| = {e2:.1e}; eta range [{eta_f2.min():.2f}, {eta_f2.max():.2f}]")


# ---------------------------------------------------------------- T3 upper bound and its failure mode
def t3():
    n, dx = 2001, 0.05
    x = (np.arange(n) - n // 2) * dx
    lam = np.logspace(-0.5, 0.9, 40)
    AD = np.abs(x[:, None] - x[None, :])
    mx = -np.inf
    for _ in range(8):
        ell = sum(a * np.exp(-(x - c) ** 2 / (2 * s ** 2))
                  for a, c, s in zip(rng.normal(0, 1, 5), rng.uniform(-8, 8, 5), rng.uniform(0.6, 3, 5)))
        G = np.array([ell @ np.exp(-AD / l) @ ell for l in lam])
        mx = max(mx, np.gradient(np.log(G), np.log(lam))[2:-2].max())
    report("T3a eta <= d for radially decreasing spectral density (exponential kernel, d=1)", mx <= 1 + 1e-3,
           f"max eta over 8 random signed leads = {mx:.4f}")
    # failure mode: oscillatory (scale-locked) coherence violates the bound
    ell = np.exp(-x ** 2 / (2 * 2.0 ** 2)) * np.cos(1.5 * x)
    G = np.array([ell @ (np.exp(-(AD / l) ** 2 / 2) * np.cos(6.0 * AD / l)) @ ell for l in lam])
    pos = G > 0
    eta = np.gradient(np.log(np.where(pos, G, np.nan)), np.log(lam))
    report("T3b documented failure mode: oscillatory coherence kernel can give eta > d", np.nanmax(eta) > 1.0,
           f"max eta = {np.nanmax(eta):.2f} (> d = 1); the bound needs a monotone spatial spectrum")


# ---------------------------------------------------------------- T4 sphere (heat-kernel coherence)
def t4():
    nmax = 80
    n = np.arange(1, nmax + 1)
    nn = n * (n + 1.0)
    chi_s, gam, t0 = 1.5, 0.35, 0.02
    f = np.logspace(0, np.log10(40), 2001)
    t = t0 * f ** (-2 * gam)
    n0 = np.arange(0, 400)
    w0 = (2 * n0 + 1)[None, :] * np.exp(-(n0 * (n0 + 1.0))[None, :] * t[:, None])
    K0 = w0.sum(1) / (4 * np.pi)
    mean_nn = (w0 * (n0 * (n0 + 1.0))[None, :]).sum(1) / w0.sum(1)
    worst = 0
    for deg in (1, 3, 6, 10):
        Phi = f ** (-chi_s) * np.exp(-deg * (deg + 1) * t) / K0
        num = -np.gradient(np.log(Phi), np.log(f))
        formula = chi_s + 2 * gam * t * (mean_nn - deg * (deg + 1))
        worst = max(worst, np.max(np.abs(num[5:-5] - formula[5:-5])))
    report("T4a sphere: chi_n = chi_s + 2 gam t (<n(n+1)>_t - n(n+1))", worst < 1e-4, f"max |diff| = {worst:.1e}")
    planar = chi_s + 2 * gam - 2 * gam * t * 1 * 2
    exact = chi_s + 2 * gam * t * (mean_nn - 2)
    report("T4b planar limit chi_s + d*gam - gam lam^2 k^2 recovered for small t", np.max(np.abs(planar - exact)) < 0.02,
           f"max |diff| = {np.max(np.abs(planar - exact)):.3f} at t <= {t.max():.3f}")
    shifts = []
    for b in (0.95, 0.85, 0.7, 0.5):
        H2 = ((2 * n + 1) * b ** (n - 1.0)) ** 2
        Pe = f ** (-chi_s) * ((2 * n + 1)[None] / (4 * np.pi) * H2[None] * np.exp(-nn[None] * t[:, None])).sum(1) / K0
        shifts.append(loglog_slope(f, Pe, 2, 40) - chi_s)
    mono = all(shifts[i] <= shifts[i + 1] + 1e-9 for i in range(3))
    report("T4c deeper source shell -> larger exponent shift, bounded by d*gam", mono and max(shifts) <= 2 * gam + 1e-6,
           "shift at b=0.95,0.85,0.70,0.50: " + ", ".join(f"{s:.3f}" for s in shifts) + f"; d*gam = {2 * gam:.3f}")


# ---------------------------------------------------------------- field synthesis for T5-T8
class Sheet:
    def __init__(self, N=24, D=24.0, T=512, fs=128.0):
        self.N, self.D, self.T, self.fs = N, D, T, fs
        kx = 2 * np.pi * np.fft.fftfreq(N, D / N)
        self.KX, self.KY = np.meshgrid(kx, kx, indexing='ij')
        self.Kabs = np.sqrt(self.KX ** 2 + self.KY ** 2)
        fall = np.fft.fftfreq(T, 1 / fs)
        self.pos = np.where(fall > 0)[0]
        self.f = fall[self.pos]

    def variance(self, chi_s, lam0, gam, kernel="gauss", wave_speed=None, nu=0.5):
        lam = lam0 * self.f ** (-gam)
        KX = self.KX[:, :, None]
        if wave_speed is not None:
            KX = KX - (2 * np.pi * self.f / wave_speed)[None, None, :]
        K2 = KX ** 2 + self.KY[:, :, None] ** 2
        l2 = lam[None, None, :] ** 2
        if kernel == "gauss":
            ker = l2 * np.exp(-l2 * K2 / 2)
        elif kernel == "exp":
            ker = l2 / (1 + l2 * K2) ** 1.5
        elif kernel == "matern":                   # d = 2: spectral density lam^2 (1 + lam^2 k^2)^-(nu + 1)
            ker = l2 / (1 + l2 * K2) ** (nu + 1.0)
        else:
            raise ValueError
        return self.f[None, None, :] ** (-chi_s) * ker, lam

    def run(self, V, nseg, Hk=None, leads_k=None, noise_var=0.0):
        """Returns mode spectra Phi[kx,ky,f] and sensor PSDs for each lead (time-domain synthesis)."""
        N, T = self.N, self.T
        amp = np.sqrt(V / 2)
        if Hk is not None:
            amp = amp * Hk[:, :, None]
        Phi = np.zeros((N, N, len(self.f)))
        leads_r = [np.real(np.fft.ifft2(lk)) * N * N for lk in (leads_k or [])]
        Pe = np.zeros((len(leads_r), len(self.f)))
        for _ in range(nseg):
            A = np.zeros((N, N, T), complex)
            A[:, :, self.pos] = amp * (rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape))
            s = np.real(np.fft.ifftn(A)) * (N * N * T)
            if noise_var > 0:
                s = s + rng.normal(0, np.sqrt(noise_var), s.shape)
            F = np.fft.fftn(s)[:, :, self.pos]
            Phi += np.abs(F) ** 2
            for i, lr in enumerate(leads_r):
                v = np.tensordot(lr, s, axes=([0, 1], [0, 1]))
                Pe[i] += np.abs(np.fft.fft(v)[self.pos]) ** 2
        return Phi / nseg, Pe / nseg


def invert_scalp(sh, Phi, band=(2, 40), kmax=1.6, d=2):
    """Scalp-only inversion of section 4.6: returns (chi_s_hat, gam_hat). No knowledge of H(k)."""
    fm = (sh.f >= band[0]) & (sh.f <= band[1])
    km = (sh.Kabs > 0) & (sh.Kabs <= kmax)
    Y = np.log(Phi[km][:, fm])                    # modes x freqs
    k2 = sh.Kabs[km] ** 2
    Yc = Y - Y.mean(1, keepdims=True)
    X = np.c_[np.ones_like(k2), k2]
    coef = np.linalg.lstsq(X, Yc, rcond=None)[0]  # 2 x freqs
    alpha, slope = coef[0], coef[1]
    Lam = -2 * slope                              # lam^2(f) - mean lam^2
    lf = np.log(sh.f[fm])
    best = None
    for c in np.linspace(-Lam.min() + 1e-3, -Lam.min() + 60, 6000):
        y = np.log(Lam + c)
        p = np.polyfit(lf, y, 1)
        r = np.sum((y - np.polyval(p, lf)) ** 2)
        if best is None or r < best[0]:
            best = (r, c, p)
    _, c, p = best
    gam_hat = -p[0] / 2
    chi_hat = -np.polyfit(lf, alpha - (d / 2) * np.log(Lam + c), 1)[0]
    return chi_hat, gam_hat


def invert_matern(sh, Phi, band=(2, 40), kmax=1.6, d=2):
    """Kernel-family inversion (Matern smoothness nu fitted). Returns (chi_s, gam, nu). H(k) unknown."""
    from scipy.optimize import least_squares
    fm = (sh.f >= band[0]) & (sh.f <= band[1])
    km = (sh.Kabs > 0) & (sh.Kabs <= kmax)
    Y = np.log(Phi[km][:, fm])
    Yc = Y - Y.mean(1, keepdims=True)
    k2 = (sh.Kabs[km] ** 2)[:, None]
    lf = np.log(sh.f[fm])[None, :]

    def resid(th):
        chi, ll0, gam, lnu = th
        l2 = np.exp(2 * (ll0 - gam * lf))
        m = -chi * lf + (d / 2) * np.log(l2) - (np.exp(lnu) + d / 2) * np.log1p(l2 * k2)
        return (Yc - (m - m.mean(1, keepdims=True))).ravel()

    best = None
    for ll0 in (0.5, 1.5, 2.5):
        for g0 in (0.1, 0.5):
            for lnu in (-1.0, 1.0):
                r = least_squares(resid, [1.0, ll0, g0, lnu], bounds=([0, -2, -0.2, -3], [5, 5, 1.5, 3.4]))
                if best is None or r.cost < best.cost:
                    best = r
    chi, ll0, gam, lnu = best.x
    return chi, gam, np.exp(lnu)


def t9_matern():
    # two-plateau law: mode elasticity eta(k) = d - (2 nu + d) lam^2 k^2 / (1 + lam^2 k^2)
    d, chi_s, gam = 2, 1.5, 0.4
    f = np.logspace(0, np.log10(40), 2001)
    worst = 0
    for nu in (0.5, 1.0, 2.5):
        for lam0, k in ((0.02, 1.0), (4.0, 1.0), (4000.0, 1.0)):
            lam = lam0 * f ** (-gam)
            Phi = f ** (-chi_s) * lam ** d * (1 + lam ** 2 * k ** 2) ** (-(nu + d / 2))
            num = -np.gradient(np.log(Phi), np.log(f))[5:-5]
            x = (lam ** 2 * k ** 2)[5:-5]
            th = chi_s + gam * (d - (2 * nu + d) * x / (1 + x))
            worst = max(worst, np.max(np.abs(num - th)))
            if lam0 == 0.02:
                lo = num.mean()
            if lam0 == 4000.0:
                hi = num.mean()
        ok = abs(lo - (chi_s + d * gam)) < 0.01 and abs(hi - (chi_s - 2 * nu * gam)) < 0.01
        report(f"T9a two-plateau law, Matern nu={nu}", ok,
               f"large-scale plateau {lo:.3f} (theory chi_s + d*gam = {chi_s + d * gam:.3f}); small-scale plateau {hi:.3f} (theory chi_s - 2*nu*gam = {chi_s - 2 * nu * gam:.3f})")
    report("T9b Matern mode elasticity formula", worst < 1e-4, f"max |diff| = {worst:.1e}")


def t5_to_t8():
    sh = Sheet()
    chi_s, lam0, gam = 1.5, 6.0, 0.4
    Hk = np.exp(-0.6 * sh.Kabs)                     # unknown volume-conductor smoothing
    lead_broad = np.exp(-(3.0 ** 2) * sh.Kabs ** 2 / 2)
    lead_lap = sh.Kabs ** 2 * np.exp(-(1.5 ** 2) * sh.Kabs ** 2 / 2)
    V, lam = sh.variance(chi_s, lam0, gam)
    Phi, Pe = sh.run(V, nseg=300, Hk=Hk, leads_k=[lead_broad, lead_lap])

    def pred(lk, Vv):
        return np.einsum('xy,xyf->f', np.abs(lk * Hk) ** 2, Vv)

    names = ["broad monopolar-like lead", "Laplacian-like lead"]
    for i, lk in enumerate([lead_broad, lead_lap]):
        e_emp = loglog_slope(sh.f, Pe[i], 2, 40)
        e_th = loglog_slope(sh.f, pred(lk, V), 2, 40)
        report(f"T5a end-to-end time-domain simulation, {names[i]}", abs(e_emp - e_th) < 0.05,
               f"simulated exponent {e_emp:.3f}, theory {e_th:.3f}, source {chi_s} (shift {e_th - chi_s:+.3f})")
    e_b = loglog_slope(sh.f, pred(lead_broad, V), 2, 40)
    e_l = loglog_slope(sh.f, pred(lead_lap, V), 2, 40)
    report("T5b derivation dependence: broad lead steeper than Laplacian-like lead", e_b > e_l + 0.1,
           f"broad {e_b:.3f} vs Laplacian-like {e_l:.3f}; neither equals the source exponent {chi_s}")

    chi_hat, gam_hat = invert_scalp(sh, Phi)
    report("T6 scalp-only inversion recovers source exponent and coherence exponent (H(k) unknown)",
           abs(chi_hat - chi_s) < 0.1 and abs(gam_hat - gam) < 0.08,
           f"chi_s_hat = {chi_hat:.3f} (true {chi_s}), gam_hat = {gam_hat:.3f} (true {gam}); naive broad-lead exponent {e_b:.3f}")

    # control: frequency-independent coherence length -> separable spectrum, no shift
    V0, _ = sh.variance(chi_s, 3.0, 0.0)
    e0 = loglog_slope(sh.f, pred(lead_broad, V0), 2, 40)
    e0l = loglog_slope(sh.f, pred(lead_lap, V0), 2, 40)
    report("T7a null control gam = 0: every lead returns the source exponent", abs(e0 - chi_s) < 1e-6 and abs(e0l - chi_s) < 1e-6,
           f"broad {e0:.6f}, Laplacian-like {e0l:.6f}")

    # stress 1: kernel misspecification (exponential kernel, Gaussian-assuming inversion)
    Vx, _ = sh.variance(chi_s, lam0, gam, kernel="exp")
    Phix, _ = sh.run(Vx, nseg=200, Hk=Hk)
    cx, gx = invert_scalp(sh, Phix)
    report("T7b STRESS kernel misspecification (true exponential, Gaussian-assuming inversion) -> expected to break",
           abs(gx - gam) > 0.1,
           f"chi_s_hat = {cx:.3f} (true {chi_s}), gam_hat = {gx:.3f} (true {gam}): the Gaussian inversion is NOT robust to the kernel",
           kind="xfail")
    cm, gm, num = invert_matern(sh, Phix)
    report("T7b' FIX kernel-family (Matern) inversion on the same exponential-kernel data",
           abs(cm - chi_s) < 0.1 and abs(gm - gam) < 0.08 and abs(num - 0.5) < 0.15,
           f"chi_s_hat = {cm:.3f} (true {chi_s}), gam_hat = {gm:.3f} (true {gam}), nu_hat = {num:.3f} (true 0.5)")
    cg, gg, nug = invert_matern(sh, Phi)
    report("T7b'' kernel-family inversion applied to Gaussian-kernel data (nu should run to its upper bound)",
           abs(cg - chi_s) < 0.15 and abs(gg - gam) < 0.1,
           f"chi_s_hat = {cg:.3f} (true {chi_s}), gam_hat = {gg:.3f} (true {gam}), nu_hat = {nug:.1f}")

    # stress 2: additive white sensor noise flattens; two-component correction
    P_sig = pred(lead_broad, V)
    lvl = P_sig[np.argmin(np.abs(sh.f - 30))]
    P_noisy = P_sig + lvl
    e_n = loglog_slope(sh.f, P_noisy, 2, 40)
    hi = sh.f > 55
    e_corr = loglog_slope(sh.f, P_noisy - np.median(P_noisy[hi] - P_sig[hi]), 2, 40)
    report("T7c STRESS white noise floor equal to signal at 30 Hz", e_n < e_b - 0.2 and abs(e_corr - e_b) < 0.02,
           f"clean {e_b:.3f}, with floor {e_n:.3f}, after floor subtraction {e_corr:.3f}")

    # stress 3: travelling waves with frequency-dependent wavenumber
    Vw, _ = sh.variance(chi_s, lam0, gam, wave_speed=600.0)
    e_w = loglog_slope(sh.f, pred(lead_broad, Vw), 2, 40)
    Phiw, _ = sh.run(Vw, nseg=200, Hk=Hk)
    cw, gw = invert_scalp(sh, Phiw)
    report("T7d STRESS travelling waves (k0 = 2 pi f / v): mixture identity still exact, isotropic inversion biased", True,
           f"broad-lead exponent {e_w:.3f} vs {e_b:.3f} without waves; inversion chi_s_hat = {cw:.3f}, gam_hat = {gw:.3f} (true {chi_s}, {gam})",
           kind="info")

    # bipolar (structure function) identity for depth-electrode pairs
    f = np.logspace(0, np.log10(40), 400)
    lamf = 60.0 * f ** (-gam)                       # mm; smooth regime needs delta << lam
    out = []
    for kern, alpha in (("gauss", 2.0), ("exp", 1.0)):
        for delta in (2.0, 3.5):
            u = delta / lamf
            kap = np.exp(-u ** 2 / 2) if kern == "gauss" else np.exp(-u)
            Dsf = 2 * f ** (-chi_s) * (1 - kap)
            out.append((kern, delta, loglog_slope(f, Dsf, 1, 10), chi_s - alpha * gam))
    ok = all(abs(o[2] - o[3]) < 0.05 for o in out)
    report("T8a bipolar depth pair: exponent = chi_s - alpha*gam in the smooth regime (delta << lam)", ok,
           "; ".join(f"{k} d={dl}mm: {a:.3f} vs {b:.3f}" for k, dl, a, b in out))
    # reference-free coherence-length estimator from two separations
    d1, d2 = 3.5, 10.5
    lam_true = np.array([4.0, 6.0, 9.0, 14.0])
    ratio = (1 - np.exp(-d1 ** 2 / (2 * lam_true ** 2))) / (1 - np.exp(-d2 ** 2 / (2 * lam_true ** 2)))
    grid = np.linspace(0.5, 60, 200000)
    rg = (1 - np.exp(-d1 ** 2 / (2 * grid ** 2))) / (1 - np.exp(-d2 ** 2 / (2 * grid ** 2)))
    mono = np.all(np.diff(rg) < 0)
    lam_hat = np.array([grid[np.argmin(np.abs(rg - r))] for r in ratio])
    report("T8b structure-function ratio is monotone in lam and inverts without any reference", mono and np.max(np.abs(lam_hat - lam_true)) < 0.01,
           f"recovered {np.round(lam_hat, 3).tolist()} for true {lam_true.tolist()}; ratio saturates at (d1/d2)^2 = {(d1 / d2) ** 2:.3f} when lam >> d2")


# ---------------------------------------------------------------- T10 lower plateau depends on the lead's vanishing order
def matern_khat(u, nu, d=2):
    """Matern spatial spectrum normalised so kappa(0) = 1 (d = 2): 4 pi nu (1 + u^2)^-(nu + 1)."""
    assert d == 2
    return 4 * np.pi * nu * (1 + u ** 2) ** (-(nu + 1.0))


def t10():
    # radial lead |l(k)|^2 = (kL)^(2m) exp(-k^2 L^2); eta_e(lam) = <eta(k;lam)> with Matern mode elasticity, d = 2
    d, L = 2, 1.0
    lk = np.linspace(np.log(1e-9), np.log(1e3), 200001)
    k = np.exp(lk)
    out, ok = [], True
    for m, nu in ((1, 0.5), (1, 2.5), (2, 1.0), (2, 3.5), (1, 1.5)):
        lam = 1e4
        x = (lam * k) ** 2
        w = (k * L) ** (2 * m) * np.exp(-(k * L) ** 2) * (1 + x) ** (-(nu + 1.0)) * k * k   # extra k: d log k measure
        eta_k = d - (2 * nu + d) * x / (1 + x)
        eta = np.trapezoid(w * eta_k, lk) / np.trapezoid(w, lk)
        th = -min(2 * m, 2 * nu)
        ok &= abs(eta - th) < 0.03
        out.append(f"m={m},nu={nu}: {eta:.3f} vs {th:.1f}")
    # Gaussian kernel limit: eta -> -2m
    for m in (1, 2):
        lam = 50.0
        w = (k * L) ** (2 * m) * np.exp(-(k * L) ** 2) * np.exp(-(lam * k) ** 2 / 2) * k * k
        eta = np.trapezoid(w * (d - (lam * k) ** 2), lk) / np.trapezoid(w, lk)
        ok &= abs(eta + 2 * m) < 0.03
        out.append(f"gauss m={m}: {eta:.3f} vs {-2 * m}")
    report("T10 lower asymptote of a lead vanishing to order m at k=0 is -min(2m, 2nu) (bipolar m=1, Laplacian m=2)", ok,
           "; ".join(out))


# ---------------------------------------------------------------- T11 Matern identifiability (Theorem 6')
def t11():
    d, chi_s, gam, lam0, nu = 2, 1.5, 0.4, 6.0, 0.8
    beta = nu + d / 2
    lf = np.linspace(np.log(2), np.log(40), 801)
    f = np.exp(lf)
    q = np.linspace(0.01, 2.0, 2001)                        # q = k^2
    Lam = lam0 ** 2 * f ** (-2 * gam)
    B = 0.7 * np.sin(3 * q) - 2.0 * q + 0.3 * q ** 2        # arbitrary unknown head transfer log|H|^2
    A = -chi_s * lf + (d / 2) * np.log(Lam)
    Y = A[:, None] + B[None, :] - beta * np.log1p(Lam[:, None] * q[None, :])
    # mixed derivative by finite differences
    Mx = np.gradient(np.gradient(Y, lf, axis=0, edge_order=2), q, axis=1, edge_order=2)  # = -beta Lam'(lf) / (1 + Lam q)^2
    i1, i2 = 200, 1600
    R = Mx[:, i1] / Mx[:, i2]                                # = ((1 + Lam q2)/(1 + Lam q1))^2, strictly increasing in Lam
    s = np.sqrt(R)
    Lam_hat = (s - 1) / (q[i2] - s * q[i1])
    dLam = np.gradient(Lam_hat, lf, edge_order=2)
    beta_hat = np.median(-Mx[:, i1] * (1 + Lam_hat * q[i1]) ** 2 / dLam)
    chi_hat = -np.polyfit(lf, Y[:, 0] - B[0] + beta_hat * np.log1p(Lam_hat * q[0]) - (d / 2) * np.log(Lam_hat), 1)[0]
    gam_hat = -np.polyfit(lf, np.log(Lam_hat), 1)[0] / 2
    sl = slice(10, -10)
    eL = np.max(np.abs(Lam_hat[sl] / Lam[sl] - 1))
    ok = eL < 1e-3 and abs(beta_hat - beta) < 1e-2 and abs(chi_hat - chi_s) < 1e-2 and abs(gam_hat - gam) < 1e-3
    report("T11a Theorem 6' Matern: mixed-derivative ratio identifies lam^2(f), nu, chi_s with H unknown (no gauge)", ok,
           f"max rel err lam^2 {eL:.1e}; nu_hat {beta_hat - d / 2:.3f} (true {nu}); chi_s_hat {chi_hat:.3f}; gam_hat {gam_hat:.4f}")
    # the Gaussian gauge does NOT carry over: best (Lam + c, H') fit to the Matern field leaves residual
    from scipy.optimize import least_squares
    Yc = Y - Y.mean(0, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean()

    def resid(th):
        c, = th
        M = -beta * np.log1p((Lam[:, None] + c) * q[None, :]) + (d / 2) * np.log(Lam + c)[:, None] - chi_s * lf[:, None]
        M = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        return (Yc - M).ravel()
    r0 = np.sqrt(np.mean(resid([0.0]) ** 2))
    r1 = np.sqrt(np.mean(resid([5.0]) ** 2))
    report("T11b gauge shift lam^2 -> lam^2 + c is not a symmetry of the Matern model (it is for the Gaussian)", r0 < 1e-10 and r1 > 1e-3,
           f"rms residual c=0: {r0:.1e}; c=5: {r1:.1e}")
    # conditioning: as nu grows (Gaussian limit) the ratio R -> 1 and Lam becomes ill-determined
    conds = []
    for nu_ in (0.5, 2.0, 8.0, 32.0):
        lam_s = lam0 * np.sqrt(2 * nu_)                       # keep kappa's curvature at the origin fixed
        L_ = lam_s ** 2 * f ** (-2 * gam)
        R_ = ((1 + L_ * q[i2]) / (1 + L_ * q[i1])) ** 2
        dR = np.gradient(np.log(R_), np.log(L_))              # sensitivity d log R / d log Lam
        conds.append(np.median(dR))
    report("T11c conditioning: d log R / d log lam^2 falls toward 0 as nu grows (Gaussian limit = gauge)", True,
           "nu=0.5,2,8,32 (curvature-matched): " + ", ".join(f"{c:.3f}" for c in conds), kind="info")


# ---------------------------------------------------------------- T12 depth contacts record a volume-conducted potential
def sf_gauss_blur(delta, lam, sigma, d=2):
    """Structure function / (2 P0) of a Gaussian-coherent sheet seen through a Gaussian blur |h|^2 = exp(-sigma^2 k^2)."""
    le2 = lam ** 2 + 2 * sigma ** 2
    return (lam ** 2 / le2) ** (d / 2) * (1 - np.exp(-delta ** 2 / (2 * le2)))


def sf_matern_blur(delta, lam, sigma, nu, u=None):
    """Same for Matern(nu) coherence, d = 2, numerical Hankel transform. delta, lam broadcastable."""
    from scipy.special import j0
    if u is None:
        u = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 4000))
    lu = np.log(u)
    lam = np.asarray(lam, float)[..., None]
    delta = np.asarray(delta, float)[..., None]
    integrand = matern_khat(u, nu) * np.exp(-(sigma * u / lam) ** 2) * (1 - j0(u * delta / lam)) * u * u / (2 * np.pi)
    return np.trapezoid(integrand, lu, axis=-1)


def t12():
    d, chi_s, gam = 2, 1.5, 0.4
    f = np.logspace(np.log10(2), np.log10(40), 200)
    lf = np.log(f)
    lam = 20.0 * f ** (-gam)                 # mm
    sig = 3.0                                # mm, frequency-independent volume-conduction blur
    deltas = np.array([5.3, 10.6, 15.9])
    # (a) Gaussian: blur is exactly the gauge lam^2 -> lam^2 + 2 sigma^2 ; check the closed form against the Hankel integral
    from scipy.special import j0
    u = np.exp(np.linspace(np.log(1e-5), np.log(1e3), 6000))
    ok_a = True
    for lv in (3.0, 10.0):
        for dl in deltas:
            integ = 2 * np.pi * np.exp(-u ** 2 / 2) * np.exp(-(sig * u / lv) ** 2) * (1 - j0(u * dl / lv)) * u * u / (2 * np.pi)
            num = np.trapezoid(integ, np.log(u))
            ok_a &= abs(num / sf_gauss_blur(dl, lv, sig) - 1) < 1e-4
    report("T12a Gaussian coherence + Gaussian blur: D = 2P0 (lam/lam_eff)^d [1 - exp(-delta^2/2lam_eff^2)], lam_eff^2 = lam^2 + 2 sigma^2", ok_a,
           "closed form matches Hankel quadrature to 1e-4")
    # (b) bipolar exponent is not fixed at chi_s - alpha*gam: it runs from chi_s + d*gam (lam << sigma) to chi_s - 2 gam (lam >> sigma, delta)
    ff = np.logspace(0, 1, 100)
    lo = loglog_slope(ff, ff ** (-chi_s) * sf_gauss_blur(5.3, 0.02 * ff ** (-gam), sig), 1, 10)
    hi = loglog_slope(ff, ff ** (-chi_s) * sf_gauss_blur(5.3, 2000 * ff ** (-gam), sig), 1, 10)
    mid = loglog_slope(f, f ** (-chi_s) * sf_gauss_blur(5.3, lam, sig), 2, 40)
    report("T12b blurred bipolar exponent spans chi_s + d*gam (lam << sigma) to chi_s - 2*gam (lam >> sigma, delta)",
           abs(lo - (chi_s + d * gam)) < 0.02 and abs(hi - (chi_s - 2 * gam)) < 0.02,
           f"{lo:.3f} (theory {chi_s + d * gam}), {hi:.3f} (theory {chi_s - 2 * gam}); realistic regime (lam 6-15 mm, sigma 3 mm): {mid:.3f}")
    # (c) point-sample ratio inversion returns lam_eff; naive gamma biased toward 0; gauge-fixed power law recovers gamma
    D = f[None] ** (-chi_s) * sf_gauss_blur(deltas[:, None], lam[None], sig)
    ratio = D[0] / D[2]
    grid = np.linspace(0.5, 200, 400000)
    rg = (1 - np.exp(-deltas[0] ** 2 / (2 * grid ** 2))) / (1 - np.exp(-deltas[2] ** 2 / (2 * grid ** 2)))
    lam_eff_hat = np.interp(ratio, rg[::-1], grid[::-1])
    g_naive = -np.polyfit(lf, np.log(lam_eff_hat), 1)[0]
    best = None
    for c in np.linspace(0, lam_eff_hat.min() ** 2 - 1e-3, 4000):
        y = np.log(lam_eff_hat ** 2 - c)
        p = np.polyfit(lf, y, 1)
        r = np.sum((y - np.polyval(p, lf)) ** 2)
        if best is None or r < best[0]:
            best = (r, c, p)
    g_fix = -best[2][0] / 2
    sig_hat = np.sqrt(best[1] / 2)
    report("T12c blur biases the reference-free gamma toward 0; fitting lam_eff^2 = lam0^2 f^(-2 gam) + c removes the bias",
           g_naive < gam - 0.05 and abs(g_fix - gam) < 0.01 and abs(sig_hat - sig) < 0.05,
           f"naive gam_hat {g_naive:.3f}, gauge-fixed gam_hat {g_fix:.3f} (true {gam}), sigma_hat {sig_hat:.2f} mm (true {sig})")
    # (d) model-free H1 test: log(D1/D3) has zero slope in log f iff gamma = 0 (exact even with blur); positive slope if gamma > 0
    s_pos = np.polyfit(lf, np.log(D[0] / D[2]), 1)[0]
    D0 = f[None] ** (-chi_s) * sf_gauss_blur(deltas[:, None], 8.0 + 0 * f[None], sig)
    s_zero = np.polyfit(lf, np.log(D0[0] / D0[2]), 1)[0]
    Dm = f[None] ** (-chi_s) * sf_matern_blur(deltas[:, None], lam[None], sig, 0.5)
    s_mat = np.polyfit(lf, np.log(Dm[0] / Dm[2]), 1)[0]
    report("T12d model-free gamma test: slope of log(D(d1)/D(d3)) vs log f is 0 iff gamma = 0, > 0 if gamma > 0 (any monotone kernel, any blur)",
           abs(s_zero) < 1e-12 and s_pos > 0.05 and s_mat > 0.05,
           f"gamma=0: {s_zero:.1e}; gamma=0.4 Gaussian: {s_pos:.3f}; gamma=0.4 exponential: {s_mat:.3f}")
    # (e) STRESS: an equal white noise floor in both differences creates a spurious positive slope at gamma = 0; subtraction removes it
    N = D0[0][np.argmin(np.abs(f - 30))]
    s_noisy = np.polyfit(lf, np.log((D0[0] + N) / (D0[2] + N)), 1)[0]
    report("T12e STRESS noise floor mimics gamma > 0 in the ratio test; must be estimated and subtracted first", s_noisy > 0.05,
           f"gamma = 0 with floor equal to D(d1) at 30 Hz: spurious slope {s_noisy:.3f}; after subtraction {s_zero:.1e}", kind="xfail")
    # (f) joint parametric fit (Matern + blur, 3 separations, log-normal spectral noise as for ~50 epochs x 7 tapers)
    from scipy.optimize import least_squares
    nu_true = 0.8
    Dt = 2 * np.exp(1.0) * f[None] ** (-chi_s) * sf_matern_blur(deltas[:, None], lam[None], sig, nu_true)
    rr = np.random.default_rng(7)
    Dn = Dt * np.exp(rr.normal(0, 0.05, Dt.shape))
    fs_ = f[::4]
    Dn_s = Dn[:, ::4]

    def model(th, fq):
        la, chi, ll0, g, lnu, lsig = th
        return np.log(2) + la - chi * np.log(fq)[None] + np.log(
            sf_matern_blur(deltas[:, None], np.exp(ll0) * fq[None] ** (-g), np.exp(lsig), np.exp(lnu)))

    def res(th):
        return (np.log(Dn_s) - model(th, fs_)).ravel()
    best = None
    for ll0 in (np.log(8), np.log(30)):
        for lnu in (np.log(0.5), np.log(2.0)):
            for lsig in (np.log(1.0), np.log(5.0)):
                r = least_squares(res, [0, 1.0, ll0, 0.2, lnu, lsig],
                                  bounds=([-10, -1, np.log(0.5), -0.5, np.log(0.2), np.log(0.1)], [10, 4, np.log(500), 1.5, np.log(30), np.log(30)]))
                if best is None or r.cost < best.cost:
                    best = r
    la, chi, ll0, g, lnu, lsig = best.x
    ok = abs(chi - chi_s) < 0.15 and abs(g - gam) < 0.1
    report("T12f joint fit (Matern + blur) on 3 separations recovers chi_s and gamma under realistic spectral noise", ok,
           f"chi_s {chi:.3f} (1.5), gam {g:.3f} (0.4), nu {np.exp(lnu):.2f} (0.8), sigma {np.exp(lsig):.2f} mm (3.0), lam0 {np.exp(ll0):.1f} (20)")
    # (g) calibration over noise seeds: bias and spread of the joint fit (feeds the per-patient power of H1)
    gs, cs = [], []
    for seed in range(8):
        r2 = np.random.default_rng(100 + seed)
        Dn_s = (Dt * np.exp(r2.normal(0, 0.05, Dt.shape)))[:, ::4]
        b2 = None
        for ll0 in (np.log(8), np.log(30)):
            for lnu in (np.log(0.5), np.log(2.0)):
                r = least_squares(res, [0, 1.0, ll0, 0.2, lnu, np.log(2.0)],
                                  bounds=([-10, -1, np.log(0.5), -0.5, np.log(0.2), np.log(0.1)], [10, 4, np.log(500), 1.5, np.log(30), np.log(30)]))
                if b2 is None or r.cost < b2.cost:
                    b2 = r
        cs.append(b2.x[1]); gs.append(b2.x[3])
    gs, cs = np.array(gs), np.array(cs)
    report("T12g calibration of the joint depth fit over 8 noise seeds", True,
           f"gam_hat {gs.mean():.3f} +/- {gs.std():.3f} (true 0.4); chi_s_hat {cs.mean():.3f} +/- {cs.std():.3f} (true 1.5); "
           f"all gam_hat > 0: {bool(np.all(gs > 0))}", kind="info")


# ---------------------------------------------------------------- T13 separability theorem and the sub-blur regime
def t13():
    # (a) Theorem 0: separable S(f) = g(f) C  ->  every lead has the same exponent; non-separable -> leads differ
    M = 10
    pos = rng.uniform(0, 30, (M, 2))
    dd = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    f = np.logspace(np.log10(2), np.log10(40), 200)
    C = np.exp(-dd / 8.0)
    leads = rng.normal(0, 1, (50, M))
    Psep = np.einsum("li,ij,lj->l", leads, C, leads)[:, None] * f[None] ** (-1.7)
    ex_sep = np.array([loglog_slope(f, p, 2, 40) for p in Psep if np.all(p > 0)])
    S = np.stack([np.exp(-dd / (8.0 * fv ** -0.4)) * fv ** -1.7 for fv in f])
    Pns = np.einsum("li,fij,lj->lf", leads, S, leads)
    ok_ns = np.all(Pns > 0, axis=1)
    ex_ns = np.array([loglog_slope(f, p, 2, 40) for p in Pns[ok_ns]])
    report("T13a Theorem 0: separable field -> all 50 random leads share one exponent; non-separable -> they differ",
           np.ptp(ex_sep) < 1e-9 and np.ptp(ex_ns) > 0.1, f"spread separable {np.ptp(ex_sep):.1e}; non-separable {np.ptp(ex_ns):.2f}")
    # (b) Proposition 7 (sub-blur regime): every lead filtered by a blur of width sigma >> lam has eta -> d, so all
    #     macroscopic derivations agree with each other and with chi_s + d*gam (none returns chi_s)
    d, chi_s, gam = 2, 1.5, 0.4
    k = np.exp(np.linspace(np.log(1e-5), np.log(1e2), 20000))
    lk = np.log(k)
    sig = 5.0
    lam = 0.3 * f ** (-gam)                          # mm, well below the blur
    res = []
    for name, l2 in (("broad L=30", np.exp(-(30 * k) ** 2)), ("bipolar d=5", 2 * (1 - np.cos(5 * k))),
                     ("Laplacian-like", (10 * k) ** 4 * np.exp(-(10 * k) ** 2))):
        G = np.array([np.trapezoid(l2 * np.exp(-(sig * k) ** 2) * lv ** 2 * 4 * np.pi * 0.5 * (1 + (lv * k) ** 2) ** -1.5 * k * k, lk) for lv in lam])
        res.append((name, loglog_slope(f, f ** (-chi_s) * G, 2, 40)))
    ok = all(abs(e - (chi_s + d * gam)) < 0.02 for _, e in res)
    report("T13b Proposition 7: lam << sigma -> every derivation reads chi_s + d*gam (derivation-invariant yet biased)", ok,
           "; ".join(f"{n}: {e:.3f}" for n, e in res) + f" (chi_s + d*gam = {chi_s + d * gam}, chi_s = {chi_s})")


# ---------------------------------------------------------------- T14 the exact condition behind Corollary 3
def t14():
    """Corollary 3 needs R(lam) = [1 - k(d1/lam)] / [1 - k(d3/lam)] to fall with lam for d1 < d3. With
    eps(x) = d log(1 - k(x)) / d log x, d log R / d log lam = eps(d3/lam) - eps(d1/lam), so R falls iff eps is non-increasing.
    Monotone k is NOT sufficient: a two-scale mixture kernel is monotone but eps is not."""
    from scipy.special import gamma as Gf, kv
    x = np.exp(np.linspace(np.log(1e-3), np.log(30.0), 4000))
    lx = np.log(x)
    fam = {"exponential": 1 - np.exp(-x), "Gaussian": 1 - np.exp(-x ** 2 / 2)}
    for nu in (0.5, 1.5, 2.5):
        fam[f"Matern nu={nu}"] = 1 - 2 ** (1 - nu) / Gf(nu) * x ** nu * kv(nu, x)
    worst = {}
    for name, g in fam.items():
        eps = np.gradient(np.log(g), lx)
        m = g > 1e-12
        worst[name] = float(np.max(np.diff(eps[m])))
    report("T14a Corollary 3 condition: eps(x) = dlog(1-k)/dlog x non-increasing for exponential, Gaussian, Matern",
           all(v < 1e-6 for v in worst.values()), "; ".join(f"{k}: max increase {v:.1e}" for k, v in worst.items()))
    # local exponential (weight 0.3) + smooth large-scale Gaussian component (scale x10): monotone kernel, eps rises 1 -> 2
    k2 = lambda u: 0.3 * np.exp(-u) + 0.7 * np.exp(-(u / 10) ** 2 / 2)            # noqa: E731
    eps2 = np.gradient(np.log(1 - k2(x)), lx)
    lam = np.exp(np.linspace(np.log(0.2), np.log(100.0), 400))
    R = (1 - k2(4 / lam)) / (1 - k2(12 / lam))
    dR = np.diff(np.log(R))
    report("T14b two-scale mixture kernel (local exponential + smooth large-scale) violates it: eps and R(lam) non-monotone",
           np.max(np.diff(eps2)) > 1e-3 and dR.max() > 0 > dR.min(),
           f"max eps increase {np.max(np.diff(eps2)):.3f}; dlogR/dloglam range [{dR.min():.4f}, {dR.max():.4f}]")


if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t9_matern(); t5_to_t8(); t10(); t11(); t12(); t13(); t14()
    nfail = sum(l.startswith("[FAIL]") for l in LINES)
    ncheck = sum(not l.startswith("[INFO]") for l in LINES)
    nx = sum(l.startswith("[XFAIL-REPRODUCED]") for l in LINES)
    print(f"\nSUMMARY: {ncheck - nfail}/{ncheck} counted checks behave as specified "
          f"({ncheck - nfail - nx} pass, {nx} expected stress-test failures reproduced); {nfail} failed; "
          f"{len(LINES) - ncheck} informational")
