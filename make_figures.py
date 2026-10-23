"""
Differentiable kinematic diffraction + inverse reconstruction (PyTorch).

Idea: simulate a diffraction pattern of a small 2D crystal, add noise,
then recover the structural parameters by gradient descent through the
physics model. Also measures how reconstruction error depends on noise.

Unknowns to recover:
  a      : lattice constant (Angstrom)
  delta  : displacement of the second atom in the unit cell (fraction of a)
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ---------- 1. Forward model (kinematic diffraction) ----------
def atom_positions(a, delta, n=6):
    """n x n unit cells, 2 atoms per cell. Returns (N, 2) tensor of positions."""
    idx = torch.arange(n, dtype=torch.float32)
    ix, iy = torch.meshgrid(idx, idx, indexing="ij")
    cells = torch.stack([ix.flatten(), iy.flatten()], dim=1) * a        # (n*n, 2)
    atom1 = torch.zeros(2)
    atom2 = torch.stack([0.5 + delta, torch.tensor(0.5)]) * a
    return torch.cat([cells + atom1, cells + atom2], dim=0)

def diffraction(a, delta, q_grid, f=(1.0, 0.6), n=6):
    """Intensity |sum_j f_j exp(i q.r_j)|^2 on a grid of q vectors (M, 2)."""
    pos = atom_positions(a, delta, n)                                    # (N, 2)
    n_cells = n * n
    fj = torch.cat([torch.full((n_cells,), f[0]), torch.full((n_cells,), f[1])])
    phase = q_grid @ pos.T                                               # (M, N)
    re = (fj * torch.cos(phase)).sum(dim=1)
    im = (fj * torch.sin(phase)).sum(dim=1)
    return re**2 + im**2

def make_q_grid(qmax=4.0, npix=48):
    q = torch.linspace(-qmax, qmax, npix)
    qx, qy = torch.meshgrid(q, q, indexing="ij")
    return torch.stack([qx.flatten(), qy.flatten()], dim=1), npix

# ---------- 2. Noise (Poisson shot noise at a given electron dose) ----------
def add_noise(intensity, dose):
    """dose = average counts per pixel. Lower dose = noisier."""
    scaled = intensity / intensity.mean() * dose
    return torch.poisson(scaled) / dose * intensity.mean()

# ---------- 3. Inverse problem ----------
def reconstruct(measured, q_grid, a_init, delta_init, steps=400, lr=0.01):
    a = torch.tensor(a_init, requires_grad=True)
    delta = torch.tensor(delta_init, requires_grad=True)
    opt = torch.optim.Adam([a, delta], lr=lr)
    target = torch.log1p(measured)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.log1p(diffraction(a, delta, q_grid))
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
    return a.item(), delta.item(), loss.item()

if __name__ == "__main__":
    TRUE_A, TRUE_DELTA = 3.0, 0.08
    q_grid, npix = make_q_grid()
    clean = diffraction(torch.tensor(TRUE_A), torch.tensor(TRUE_DELTA), q_grid)

    # --- single example figure ---
    noisy = add_noise(clean, dose=20.0)
    a_hat, d_hat, _ = reconstruct(noisy, q_grid, a_init=TRUE_A * 1.02, delta_init=0.0)
    print(f"True a={TRUE_A}, delta={TRUE_DELTA} | Recovered a={a_hat:.4f}, delta={d_hat:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(clean.reshape(npix, npix).T, origin="lower"); ax[0].set_title("Clean pattern")
    ax[1].imshow(noisy.reshape(npix, npix).T, origin="lower"); ax[1].set_title("Noisy (dose=20)")
    rec = diffraction(torch.tensor(a_hat), torch.tensor(d_hat), q_grid)
    ax[2].imshow(rec.reshape(npix, npix).T, origin="lower"); ax[2].set_title("Reconstructed")
    plt.tight_layout(); plt.savefig("patterns.png", dpi=150); plt.close()

    # --- error vs noise level (the key result) ---
    doses = [200, 50, 20, 8, 3, 1]
    trials = 8
    err_a, err_d = [], []
    for dose in doses:
        ea, ed = [], []
        for _ in range(trials):
            noisy = add_noise(clean, dose)
            a_hat, d_hat, _ = reconstruct(noisy, q_grid, TRUE_A * 1.02, 0.0, steps=300)
            ea.append(abs(a_hat - TRUE_A)); ed.append(abs(d_hat - TRUE_DELTA))
        err_a.append(sum(ea) / trials); err_d.append(sum(ed) / trials)
        print(f"dose={dose:>4}: mean |err a|={err_a[-1]:.4f}  mean |err delta|={err_d[-1]:.4f}")

    plt.figure(figsize=(5, 4))
    plt.loglog(doses, err_a, "o-", label="lattice constant a (A)")
    plt.loglog(doses, err_d, "s-", label="displacement delta")
    plt.xlabel("Dose (counts/pixel)"); plt.ylabel("Mean absolute error")
    plt.legend(); plt.gca().invert_xaxis(); plt.tight_layout()
    plt.savefig("error_vs_noise.png", dpi=150)
