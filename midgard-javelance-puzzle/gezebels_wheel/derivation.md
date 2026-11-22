# Rolling wheel: derivation of (r(θ), φ(θ)) for a fixed lab point

## Problem

A wheel of radius `R` rolls without slipping to the right along the x-axis. At the start its centre is at `(0, R)`. Pick a height `0 < h ≤ 2R`. Let `(d, h)` be the point of the *lab frame* with `d > 0` that lies on the rim of the wheel at the start. This lab point stays fixed in space while the wheel rolls past it.

Attach a polar coordinate system `(r, φ)` to the wheel, centred on its axle and rotating with it. We ask: as the wheel rolls, what are the wheel-frame polar coordinates `(r(θ), φ(θ))` of the fixed lab point `(d, h)`, as a function of the rolling angle `θ`?

## Conventions

- `θ` is positive **counter-clockwise** (the usual mathematical convention).
- Rolling to the right is therefore a clockwise rotation, so the sweep runs over **negative θ**.
- Rolling without slipping: the centre moves by `R` times the angle turned, so the centre is at

  `C(θ) = (−Rθ, R)`.

  For `θ < 0` this is to the right of the start, as it should be.
- A wheel-fixed point with polar coordinates `(r, φ)` is at wheel angle `φ` when `θ = 0`. After the wheel has been rotated counter-clockwise by `θ`, that point sits at lab angle `φ + θ` measured from the centre.

So the lab position of the wheel-fixed point `(r, φ)` is

```
x = −Rθ + r cos(φ + θ)
y =   R + r sin(φ + θ)
```

(The y component is a sine, not a cosine, for the same reason the x component is a cosine: both use the same angle `φ + θ`.)

## Step 1: the starting configuration fixes d

At `θ = 0` the point is on the rim, `r = R`, at wheel angle `φ₀`:

```
d = R cos φ₀
h = R + R sin φ₀   ⇒   sin φ₀ = h/R − 1
```

Choosing `d > 0` selects `φ₀ ∈ (−π/2, π/2)`, so `cos φ₀ > 0` and

```
d = R cos φ₀ = sqrt(R² − (h − R)²) = sqrt(2Rh − h²).
```

A useful identity that is used below:

```
d² + (h − R)² = R².
```

## Step 2: the fixed-point condition

We want the wheel-frame coordinates `(r, φ)` whose lab position equals the fixed point `(d, h)`:

```
−Rθ + r cos(φ + θ) = d
 R  + r sin(φ + θ) = h
```

Rearranged:

```
r cos(φ + θ) = d + Rθ          (1)
r sin(φ + θ) = h − R           (2)
```

Equation (2) says that the vertical offset of the point from the axle is constant, `h − R`. Of course: the axle always sits at height `R`, and the point never moves. Equation (1) says the horizontal offset from the axle is `d + Rθ`, which changes linearly in `θ` because the axle translates.

In other words, in the **lab frame** the point is at offset `(d + Rθ, h − R)` from the centre. All that remains is to re-express this offset in the wheel's rotating frame.

## Step 3: r as a function of θ

Square and add (1) and (2):

```
r² = (d + Rθ)² + (h − R)²
```

so

```
r(θ) = sqrt( (d + Rθ)² + (h − R)² ).
```

## Step 4: θ as a function of r (the two branches)

Solving the same relation for `θ`:

```
(d + Rθ)² = r² − (h − R)²
d + Rθ = ± sqrt( r² − (h − R)² )

θ = ( −d ± sqrt( r² − (h − R)² ) ) / R.
```

Note the square root must be written as `(…)**(1/2)` in code; `(…)**1/2` would evaluate to half of the argument.

The two signs are two branches of the same curve:

- `+` branch: `d + Rθ ≥ 0`, i.e. `θ ≥ −d/R`. The fixed point is on the front (right-hand) side of the axle (lab x larger than the axle's).
- `−` branch: `d + Rθ ≤ 0`, i.e. `θ ≤ −d/R`. The fixed point has been passed and is behind the axle.

The branches join at `θ = −d/R`, where the axle is directly above or below the point, and `r` has its minimum `|h − R|`.

## Step 5: φ as a function of θ

From (1) and (2) the lab angle of the point as seen from the axle is

```
φ + θ = atan2( h − R, d + Rθ ).
```

Therefore

```
φ(θ) = atan2( h − R, d + Rθ ) − θ.
```

Be careful with the inverse tangent. The form `arctan( (h − R)/(d + Rθ) )` is only correct when `d + Rθ > 0`, i.e. on the `+` branch. On the `−` branch `d + Rθ < 0`, and plain `arctan` returns an angle that is off by `π`. `atan2` takes both signs into account and is correct on both branches. For `h = R` the plain `arctan` also divides by zero when `d + Rθ = 0`.

## Step 6: range of θ

The fixed point is inside the wheel only while `r ≤ R`:

```
(d + Rθ)² + (h − R)² ≤ R²
(d + Rθ)² ≤ R² − (h − R)² = d²
−d ≤ d + Rθ ≤ d
−2d/R ≤ θ ≤ 0.
```

So the point enters the wheel's rim at `θ = 0` (where `r = R`, `φ = φ₀`), passes closest to the axle at `θ = −d/R` (where `r = |h − R|`), and leaves again through the rim at `θ = −2d/R` (where `r = R`). The two rim points are mirror images across the vertical line through the axle at the moment of closest approach.

## Special cases and shape

- **`h = R`.** Then `h − R = 0` and `d = R`. The point passes exactly through the axle: `r = R |1 + θ|`, reaching `r = 0` at `θ = −1`. The angle `φ` is undefined there, and as the point goes through the centre, `φ` jumps by `π`. This is a genuine property of polar coordinates, not an error in the formulas; it is why the numerical code never queries points too close to the centre.
- **`h < R` or `h > R`.** The curve never reaches the centre; the minimum radius is `|h − R|`.
- **Shape.** Since `φ = (lab angle) − θ`, and the lab angle `φ + θ` sweeps through the range seen along a straight horizontal line at height `h − R` above the axle, while the wheel frame itself counter-rotates by `−θ`, the path in the wheel frame is a spiral-like arc. It starts and ends on the rim.

## Summary

With `d = sqrt(2Rh − h²)`, `θ ∈ [−2d/R, 0]` (negative = rolling right):

```
r(θ) = sqrt( (d + Rθ)² + (h − R)² )
φ(θ) = atan2( h − R, d + Rθ ) − θ

or equivalently, as a function of r:
θ = ( −d ± sqrt( r² − (h − R)² ) ) / R,
φ = atan2( h − R, d + Rθ ) − θ.
```

## Mapping to the numerical code

`gezebels_wheel/wheel_numeric.py` sweeps a rolling angle that is positive for clockwise rotation, so `θ_numeric = −θ_analytic`. The analytic curve parametrised by `r` is interpolated onto the numeric θ values (`gezebel_curve`), and the error is plotted against `θ / θ_max` with `θ_max = 2d/R`.

For all tested `h` (`0.25R`, `0.5R`, `0.75R`, `R`, `1.5R`) the numeric and analytic results agree to about `1e-5` in `r` and `1e-8` in `φ`. The residual error is the interpolation error of treating each polar grid cell as a bilinear quadrilateral, and it shrinks as the grid is refined.
