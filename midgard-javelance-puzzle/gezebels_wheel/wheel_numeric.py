"""Numerical check of the rolling-wheel problem (see javelance/gezebels_wheel.md).

A wheel of radius R starts with its centre at (0, R) and rolls right (clockwise
rotation theta, centre translation R*theta).  A fixed lab point (d, h) lies on
the wheel's edge at theta = 0, i.e. d = sqrt(2*R*h - h**2).

Numerical method: grid the wheel in polar coordinates (r_i, phi_j), rotate and
translate the grid into the lab for each theta, find the cell containing (d, h),
and invert that cell's bilinear map to interpolate (r, phi).

Two analytic curves are compared against it:
  * "gezebel":  the formulas in gezebels_wheel.md, implemented literally
                (parametrised by r; interpolated onto the numeric theta values)
  * "derived":  r = hypot(d - R*theta, h - R),
                phi = atan2(h - R, d - R*theta) + theta  (mod 2*pi)
"""
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

R = 1.0
H_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5]   # in units of R; h == R is allowed
NR, NPHI = 200, 720                      # grid resolution (cells)
NTHETA = 300
R_SKIP = 1e-3                            # don't query points this close to the centre


# ---------------------------------------------------------------- numerics ---
def make_grid():
    r = np.linspace(0.0, R, NR + 1)
    phi = np.linspace(0.0, 2 * np.pi, NPHI + 1)   # last column duplicates first
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    return r, phi, rr * np.cos(pp), rr * np.sin(pp)


def to_lab(x, y, theta):
    """Rotate clockwise by theta about the origin, translate centre to (R*theta, R)."""
    c, s = np.cos(theta), np.sin(theta)
    return R * theta + c * x + s * y, R - s * x + c * y


def locate(theta, r, phi, gx, gy, px, py):
    X, Y = to_lab(gx, gy, theta)
    P = [(X[:-1, :-1], Y[:-1, :-1]), (X[1:, :-1], Y[1:, :-1]),
         (X[:-1, 1:], Y[:-1, 1:]), (X[1:, 1:], Y[1:, 1:])]   # 00, 10, 01, 11
    xs = np.stack([p[0] for p in P])
    ys = np.stack([p[1] for p in P])
    cand = np.argwhere((xs.min(0) <= px) & (px <= xs.max(0)) &
                       (ys.min(0) <= py) & (py <= ys.max(0)))
    p = np.array([px, py])
    for i, j in cand:
        a, b, c, e = (np.array([q[0][i, j], q[1][i, j]]) for q in P)
        s = t = 0.5
        for _ in range(30):                      # Newton on the bilinear map
            F = (1-s)*(1-t)*a + s*(1-t)*b + (1-s)*t*c + s*t*e - p
            J = np.array([-(1-t)*a + (1-t)*b - t*c + t*e,
                          -(1-s)*a - s*b + (1-s)*c + s*e]).T
            try:
                ds, dt = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            s, t = s + ds, t + dt
            if abs(ds) + abs(dt) < 1e-13:
                break
        tol = 1e-9
        if -tol <= s <= 1 + tol and -tol <= t <= 1 + tol:
            return r[i] + s * (r[i+1] - r[i]), phi[j] + t * (phi[j+1] - phi[j])
    return None


def numeric_curve(h, grid):
    r, phi, gx, gy = grid
    d = np.sqrt(2 * R * h - h**2)
    eps = 2e-3
    thetas = np.linspace(eps, 2 * d / R - eps, NTHETA)
    out = np.full((NTHETA, 2), np.nan)
    for k, th in enumerate(thetas):
        # skip the (undefined-phi) wheel centre: distance from point to centre
        if np.hypot(d - R * th, h - R) < R_SKIP:
            continue
        res = locate(th, r, phi, gx, gy, d, h)
        if res is not None:
            out[k] = res
    return d, thetas, out[:, 0], np.mod(out[:, 1], 2 * np.pi)


# ---------------------------------------------------------------- analytics ---
def gezebel_curve(h, thetas):
    """gezebels_wheel.md (theta ccw-positive, centre moves by -R*theta), with the
    vertical offset taken as h - R.  `thetas` are the numeric (clockwise) angles, so
    the analytic theta is -thetas.  Parametrised by s = sqrt(r^2 - (h-R)^2) on both
    branches, then interpolated onto the requested thetas.
        r sin(phi+theta) = h - R,  r cos(phi+theta) = d + R*theta
        theta = (+-sqrt(r^2 - (h-R)^2) - d)/R,  phi = atan2(h-R, d+R*theta) - theta
    """
    phi0 = np.arcsin(h / R - 1)
    d = R * np.cos(phi0)
    s = np.linspace(0, d, 4001)
    r_s = np.hypot(s, h - R)
    th = np.concatenate([((-s - d) / R)[::-1], ((s - d) / R)[1:]])   # ascending
    rr = np.concatenate([r_s[::-1], r_s[1:]])
    ph = np.arctan2(h - R, d + R * th) - th
    q = -thetas
    r_an = np.interp(q, th, rr, left=np.nan, right=np.nan)
    ph_an = np.interp(q, th, ph, left=np.nan, right=np.nan)
    return r_an, np.mod(ph_an, 2 * np.pi)


def derived_curve(h, d, thetas):
    r_an = np.hypot(d - R * thetas, h - R)
    ph_an = np.arctan2(h - R, d - R * thetas) + thetas
    return r_an, np.mod(ph_an, 2 * np.pi)


def wrap(x):
    return np.angle(np.exp(1j * x))


# --------------------------------------------------------------------- main ---
def main():
    grid = make_grid()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("r(θ)", "φ(θ)  [mod 2π]", "r error (numeric − analytic)",
                        "φ error (numeric − analytic, wrapped)"))
    colors = pc.qualitative.D3
    pol = make_subplots(
        rows=3, cols=1, row_heights=[0.55, 0.225, 0.225], vertical_spacing=0.07,
        specs=[[{"type": "polar"}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=("path in wheel polar coords (r, φ): first half of θ numeric (solid), "
                        "second half analytic (dashed)",
                        "r error (numeric − analytic)", "φ error (numeric − analytic, wrapped)"))
    pol.add_trace(go.Scatterpolar(
        r=[R] * 361, theta=np.arange(361), mode="lines", name="wheel edge",
        line=dict(color="lightgray", width=3), hoverinfo="skip"), row=1, col=1)

    for n, hf in enumerate(H_VALUES):
        h = hf * R
        d, th, rn, pn = numeric_curve(h, grid)
        rg, pg = gezebel_curve(h, th)
        rd, pd_ = derived_curve(h, d, th)
        col, grp, name = colors[n % len(colors)], f"h{hf}", f"h={hf}R"
        print(f"h={hf}R: d={d:.4f}  max|dr| derived={np.nanmax(abs(rn-rd)):.2e}  "
              f"gezebel={np.nanmax(abs(rn-rg)):.2e}  "
              f"max|dphi| derived={np.nanmax(abs(wrap(pn-pd_))):.2e} gezebel={np.nanmax(abs(wrap(pn-pg))):.2e}")

        k = len(th) // 2          # first half numeric, second half analytic (shared point k)
        pol.add_trace(go.Scatterpolar(
            r=rn[:k + 1], theta=np.degrees(pn[:k + 1]), mode="lines", name=name,
            line=dict(color=col, width=4), legendgroup=grp), row=1, col=1)
        pol.add_trace(go.Scatterpolar(
            r=np.r_[rn[k], rg[k + 1:]], theta=np.degrees(np.r_[pn[k], pg[k + 1:]]),
            mode="lines", name=f"{name} analytic", line=dict(color=col, width=4, dash="dash"),
            legendgroup=grp, showlegend=False), row=1, col=1)
        t_norm = th / (2 * d / R)
        pol.add_trace(go.Scatter(
            x=t_norm, y=rn - rg, mode="lines", line=dict(color=col, width=2),
            name=name, legendgroup=grp, showlegend=False), row=2, col=1)
        pol.add_trace(go.Scatter(
            x=t_norm, y=wrap(pn - pg), mode="lines", line=dict(color=col, width=2),
            name=name, legendgroup=grp, showlegend=False), row=3, col=1)

        def add(row, y, dash, label, showleg):
            fig.add_trace(go.Scatter(
                x=th, y=y, mode="lines", line=dict(color=col, dash=dash, width=2),
                name=f"{name} {label}", legendgroup=grp, showlegend=showleg),
                row=row, col=1)

        for row, num, gz, dv in ((1, rn, rg, rd), (2, pn, pg, pd_)):
            add(row, num, "solid", "numeric", row == 1)
            add(row, gz, "dash", "gezebel", row == 1)
            add(row, dv, "dot", "derived", row == 1)
        add(3, rn - rg, "dash", "gezebel", False)
        add(3, rn - rd, "dot", "derived", False)
        add(4, wrap(pn - pg), "dash", "gezebel", False)
        add(4, wrap(pn - pd_), "dot", "derived", False)

    fig.update_xaxes(title_text="θ (rad)", row=4, col=1)
    fig.update_layout(
        height=1100, template="plotly_white",
        title="Rolling wheel: numeric (solid) vs gezebel analytic (dashed) vs derived (dotted)")
    fig.write_html("wheel_numeric.html")
    print("saved wheel_numeric.html")

    pol.update_layout(
        height=1500, template="plotly_white",
        title="Rolling wheel: polar path and numeric − analytic error")
    pol.update_polars(radialaxis=dict(range=[0, R * 1.05]),
                      angularaxis=dict(direction="counterclockwise", rotation=0))
    pol.update_xaxes(title_text="normalised θ  (θ / θ_max, 0 → 1)", row=3, col=1)
    pol.update_xaxes(title_text="normalised θ  (θ / θ_max, 0 → 1)", row=2, col=1)
    pol.update_yaxes(title_text="Δr", exponentformat="e", row=2, col=1)
    pol.update_yaxes(title_text="Δφ (rad)", exponentformat="e", row=3, col=1)
    pol.write_html("wheel_polar.html")
    print("saved wheel_polar.html")


if __name__ == "__main__":
    main()
