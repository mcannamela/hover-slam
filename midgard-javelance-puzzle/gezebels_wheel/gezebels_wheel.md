## Numerical prompt

```
Let's work on a problem on the side. Imagine I have a wheel of radius R that starts with its center at (0,R) and rolls to the right. I choose a fixed height h>0 and then find d>0 such that (d,h) is a point on the edge of
  the wheel. Then I identify each point of the wheel with a polar coordinate system (r, phi) at its center. As the wheel rotates by an angle theta and thus translates by R*theta, I want to find (r(theta), phi(theta)) ==
  (d,h). I want to do this problem completely numerically so that I can then check my analytical results. So the program should grid the wheel with polar coordinates, then apply a rotation and a translation to that grid by
  the angle theta, locate the cell that d,h occupies, and then interpolate between the grid points to get a precise value for (r,phi). We'll sweep values of theta and then plot the resulting curves.
```

## Analytical solution

sin(phi_0) = (h/R - 1)

d = R*cos(phi_0)

r*sin(phi + theta) = h - R
r*cos(phi + theta) = d + R*theta

theta = (-d +/-(r**2 - (h-R)**2)**(1/2)) / R

phi = arctan((h - R) / (d+R*theta)) - theta