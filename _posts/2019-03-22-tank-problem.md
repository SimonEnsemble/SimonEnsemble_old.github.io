---
title: ODE fun with liquid flow into tanks with complicated geometries
category: chemical engineering, process dynamics
indexing: true
comments: true
excerpt: We derive a dynamic model for the liquid level in a truncated square pyramidal tank when liquid flows in at a rate $q_i$ and is driven out of the tank by hydrostatic pressure.
author: Cory Simon
---

Liquid continuously flows into a truncated square pyramidal tank at a rate $q_i=q_i(t)$ [m$^3$/s]. Hydrostatic pressure drives flow out of the tank through a narrow pipe at its base. Via Bernoulli's equation, the volumetric flow rate out of the tank, $q$ is then proportional to $\sqrt{h}$, where $h$ [m$^3$] is the liquid level in the tank.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_1.png" alt="image" style="width: 70%;">
</figure>

The goal here is to derive a dynamic model for the liquid level $h=h(t)$ given any general input $q_i(t)$. The outlet flow rate then follows as $c\sqrt{h(t)}$. The proportionality constant $c$ is a characteristic of the fluid in the tank and the geometry of the exit line; it can be measured experimentally.

Taking the liquid as incompressible, a mass balance in the tank leads to a volume balance:

{:.centerr}
$\dfrac{dV}{dt}=q_i-c\sqrt{h}$

where $V=V(t)$ is the volume of liquid in the tank at time $t$. This differential equation is intuitive if we multiply it by $dt$ since, within such a differential time window, we can view $q_i(t)$ and $h(t)$ as constant. In a differential time span $dt$: the amount of liquid that entered the tank is $q_idt$; the amount of liquid that exited the tank is $c\sqrt{h}dt$; the change in volume $dV$ of liquid in the tank within this time span then must be $(q_i - c\sqrt{h})dt$.

The [agnostic to tank geometry] volume balance as written is not so helpful. It cannot be directly solved since $V$ is coupled to $h$ through the tank geometry, i.e., $V=V(h)$. If we know $q_i$ and $h$ at a given moment, we know the volume will change by $dV = (q_i - c\sqrt{h})dt$ after a differential time $dt$. But changing the volume will in turn change the liquid level $h$. So, to compute $dV$ at the _next_ time step, we'd need to know how $h$ changed after we incremented the volume by $dV$. This is where the tank geometry comes in.

A little more about the truncated square pyramidal tank. The view from all four sides is equivalent and is below. Its height is $H$ [m]. The top and bottom base are of length $L_t$ and $L_b$ [m], respectively. All horizontal slices through the tank reveal a square cross-section. This is a _right pyramid_, meaning that its apex (if it weren't truncated) is directly above the centroid of its base. This information will allow us to write $dV$ in terms of $dh$, making the volume balance well-posed.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_2.png" alt="image" style="width: 70%;">
</figure>

Consider when the liquid level height changes by a differential amount $dh$. This is viewed as adding a small slice of liquid whose volume is $dV$. We relate $dV$ and $dh$ through the area $A$ of the slice of liquid from the helicopter view; $dV=Adh$ since such a thin slice is approximately a rectangular prism (i.e. the area above and below are the same as $dh\rightarrow 0$). However, notably $A=A(h)$ since the area from the helicopter view gets larger for smaller liquid levels.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_3.png" alt="image" style="width: 70%;">
</figure>

Therefore our volume balance becomes:

{:.centerr}
$A(h)\dfrac{dh}{dt}=q_i-c\sqrt{h}$

and is well-posed if we can find how the area from a helicopter view varies with $h$ in our tank geometry. This is the key to solving all tank problems of this nature, regardless of the geometry.

If we define $w=w(h)$ [m] to be the length of the line that the top of the liquid makes from a side view, then the area is simply $A(h)=[w(h)]^2$.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_4.png" alt="image" style="width: 70%;">
</figure>

We can relate $w$ to $h$ and the length of the bases $L_t$ and $L_b$ if we recognize two right triangles. We bring in $L_t$ and write $w=L_t+2\theta$ with $\theta$ depicted below. So the area is $A=(L_t+2\theta)^2$.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_5.png" alt="image" style="width: 70%;">
</figure>

To determine $\theta$, we recognize two similar right triangles. At the bottom base, we decompose $L_b$ into $L_t$ plus $L_b-L_t$; half of the latter must be the base of the largest right triangle we drew. The smaller right triangle has a height of the tank $H$ minus that of the liquid $h$.

{:.centerr}
<figure>
    <img src="/images/tank_problem/tank_6.png" alt="image" style="width: 70%;">
</figure>

Because the triangles are similar, the ratio of the two sides must be the same:

{:.centerr}
$\dfrac{\theta}{H-h} = \dfrac{(L_b-L_t)/2}{H}$

allowing us to solve for $\theta$ in terms of the geometric properties of the tank and the liquid level. Plugging $\theta$ into our expression above for the area $A$ of the differential slice, we arrive at:

{:.centerr}
$A(h) = \left[\frac{h}{H} L_t + \left(1-\frac{h}{H}\right)L_b \right]^2$

The expression inside the brackets $[\cdot]^2$ is $w$. Intuitively, $w$ is a linear interpolation between $L_t$ and $L_b$. As $h\rightarrow 0$, $w$ approaches the bottom base height $L_b$. If $h \rightarrow H$, $w$ approaches the top base height $L_t$.

Finally, completing our dynamic differential equation model for the liquid level $h$ in the tank, use our derived expression $A(h)$ for our truncated square pyramidal tank:

{:.centerr}
$\left[\frac{h}{H} L_t + \left(1-\frac{h}{H}\right)L_b \right]^2 \dfrac{dh}{dt}= q_i - c\sqrt{h}$

This differential equation is non-linear. The solution $h(t)$ for a given initial condition $h(t=0)$ and input flow scheme $q_i(t)$ can be obtained (i) numerically e.g. through [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/) or (ii) via an approximate linearized model about a nominal steady state condition $\bar{h}$ and $\bar{q_i}$.

## Numerical solution

Let's write code in Julia and use DifferentialEquations.jl to numerically approximate the solution to our tank problem when the tank is initially empty and liquid flows into the tank at a constant rate.

First, we load some packages and define our parameters.

```{julia}
using DifferentialEquations
using PyPlot # for plotting
using LaTeXStrings # for LaTeX strings in plots

PyPlot.matplotlib[:style][:use]("Solarize_Light2") # dope plot style

# specify tank geometry
H = 4.0 # tank height, m
Lb = 5.0 # bottom base length, m
Lt = 2.0 # top base length, m

# specify resistance to flow
c = 1.0 # awkward units

# inlet flow rate (could be funciton of time)
qᵢ = 1.5 # m³/s

# initial liquid level
h₀ = 0.0 
```

Second, we use DifferentialEquations.jl to numerically solve the ODE.

```{julia}
tspan = (0.0, 150.0) # solve for 0 s to 150 s

# area from a helicopter view, m²
A(h, Lt, Lb, H) = (h/H * Lt + (1 - h/H) * Lb) ^ 2

# right-hand-side of ODE
rhs(h, p, t) = (qᵢ - c * sqrt(h)) / A(h, Lt, Lb, H)

# DifferentialEquations.jl syntax
prob = ODEProblem(rhs, h₀, tspan)
sol = solve(prob)
```

We can now plot the solution as follows.

```${julia}
t = range(0.0, stop=tspan[2], length=300)
h = sol.(t) # easy as that to compute solution at an array of times!

figure()
axhline(y=H, linestyle="--", color="k")
xlabel(L"$t$, time [s]")
ylabel(L"$h$, liquid level [m]")
plot(t, h, lw=3, color="orange")
```

We see how $h$ changes with time as the tank fills up. The vertical dashed line shows $H$. Despite constant flow into the tank, the tank does not overflow since hydrostatic pressure drives flow out of the tank. The liquid level reaches a steady state when the flow into the tank balances the rate at which hydrostatic pressure drives fluid out of the tank.

{:.centerr}
<figure>
    <img src="/images/tank_problem/numerical_soln.png" alt="image" style="width: 70%;">
</figure>

