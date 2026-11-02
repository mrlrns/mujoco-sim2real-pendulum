"""Is swing-up physically possible with the 3 N.m motor? Minimum-time swing-up by value iteration.

    python -m scripts.feasibility        (numpy only, about 1 minute)

Simplified model of the pendulum (rod + 1 kg bob, hinge damping 0.05, 20 ms control step, bang-bang torque
-tau, 0, +tau). For each (mass, gain) it prints the minimum number of control steps needed to bring the
pendulum from rest at the bottom to the top (|angle error| < 0.1 rad, |omega| < 1 rad/s). The episode lasts
250 steps. Inertia is approximated, and bilinear interpolation of the value function is slightly optimistic,
so treat the numbers as estimates, not as MuJoCo measurements.
"""
import numpy as np

g, L, mb, mr = 9.81, 0.6, 1.0, 0.05
Md0 = mb * L + mr * L / 2                                  # static moment of rod + bob
I0 = mb * L**2 + 0.4 * mb * 0.1**2 + mr * L**2 / 3         # inertia about the pivot (approx.)
damp, dt, sub = 0.05, 0.002, 10
th = np.linspace(-np.pi, np.pi, 361, endpoint=False); dth = th[1] - th[0]
om = np.linspace(-12, 12, 241); dom = om[1] - om[0]
TH, OM = np.meshgrid(th, om, indexing="ij")


def min_steps(mass, gear, thr=0.1, omr=1.0):
    tau_max, Md, I = 3.0 * gear, Md0 * mass, I0 * mass     # mass scales the inertia too, as in the env
    target = (abs(np.angle(np.exp(1j * (TH - np.pi)))) < thr) & (abs(OM) < omr)
    nxt = []
    for tau in (-tau_max, 0.0, tau_max):
        t, w = TH.copy(), OM.copy()
        for _ in range(sub):
            w = w + dt * (tau - Md * g * np.sin(t) - damp * w) / I
            t = t + dt * w
        t, w = np.angle(np.exp(1j * t)), np.clip(w, om[0], om[-1])
        fi = np.mod((t - th[0]) / dth, len(th)); i0 = np.floor(fi).astype(int); a = fi - i0; i1 = (i0 + 1) % len(th)
        fj = (w - om[0]) / dom; j0 = np.clip(np.floor(fj).astype(int), 0, len(om) - 2); b = fj - j0
        nxt.append((i0, i1, a, j0, b))
    V = np.full(TH.shape, 1e3); V[target] = 0
    for _ in range(600):
        best = np.full(TH.shape, 1e3)
        for i0, i1, a, j0, b in nxt:
            Vn = ((1-a)*(1-b)*V[i0, j0] + a*(1-b)*V[i1, j0] + (1-a)*b*V[i0, j0+1] + a*b*V[i1, j0+1])
            best = np.minimum(best, 1 + Vn)
        new = np.where(target, 0, best)
        if np.abs(new - V).max() < 1e-9:
            V = new; break
        V = new
    j = np.argmin(abs(om))
    return max(V[np.argmin(abs(th - 0.1)), j], V[np.argmin(abs(th + 0.1)), j])


if __name__ == "__main__":
    print(f"static gravity torque at horizontal: {Md0 * g:.2f} N.m, motor: 3.00 N.m (so no direct lift: it must pump)")
    print("mass  gain  min steps (of 250)")
    for m, gr in [(1, 1), (0.5, 1), (0.7, 1), (1.2, 1), (1.5, 1), (1, 0.7), (1, 0.5), (1.2, 0.7), (1, 2.0)]:
        print(f"{m:4.1f} {gr:5.2f}  {min_steps(m, gr):6.0f}")
