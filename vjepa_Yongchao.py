# -*- coding: utf-8 -*-

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# VJEPA + BJEPA

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from copy import deepcopy
import time

# Set global seeds for reproducibility
seed=111
torch.manual_seed(seed)
np.random.seed(seed)

# ==========================================
# 1. Environment: Decoupled Generation
# ==========================================
class DistractorSystem:
    def __init__(self, dim_s=4, dim_distractor=4, dim_x=20):
        self.dim_s = dim_s
        self.dim_d = dim_distractor
        self.dim_x = dim_x

        # Signal Dynamics (Stable Rotation)
        H = np.random.randn(dim_s, dim_s)
        Q_rot, _ = np.linalg.qr(H)
        self.A = torch.FloatTensor(Q_rot)
        self.Q_s = 0.1

        # Distractor Dynamics (Sticky Random Walk)
        self.A_distractor = 0.9 * torch.eye(dim_distractor)

        # Emission Matrices (Fixed)
        C_np = np.random.randn(dim_x, dim_s)
        C_np = C_np / np.linalg.norm(C_np, axis=0, keepdims=True)
        self.C = torch.FloatTensor(C_np)

        D_np = np.random.randn(dim_x, dim_distractor)
        D_np = D_np / np.linalg.norm(D_np, axis=0, keepdims=True)
        self.D = torch.FloatTensor(D_np)

    def generate_latent_processes(self, n_samples):
        s = torch.zeros(n_samples + 1, self.dim_s)
        s[0] = torch.randn(self.dim_s)
        d_base = torch.zeros(n_samples + 1, self.dim_d)
        d_base[0] = torch.randn(self.dim_d)

        for t in range(n_samples):
            # Signal
            s[t+1] = s[t] @ self.A.T + torch.randn(self.dim_s) * self.Q_s
            # Distractor
            d_base[t+1] = d_base[t] @ self.A_distractor.T + torch.randn(self.dim_d) * 0.3

        return s, d_base

    def observe(self, s, d_base, noise_scale):
        n_samples = s.shape[0] - 1
        d_scaled = d_base * noise_scale
        sensor_noise = torch.randn(n_samples + 1, self.dim_x) * 0.01
        x = s @ self.C.T + d_scaled @ self.D.T + sensor_noise
        return x[:-1], x[1:], s[:-1], d_scaled[:-1]

# ==========================================
# 2. Models
# ==========================================
class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc_mu = nn.Linear(input_dim, latent_dim, bias=False)
        self.enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)
        self.dec = nn.Linear(latent_dim, input_dim, bias=False)
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def forward(self, x):
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar, z

class LinearAR(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, latent_dim, bias=False)
        self.pred_dec = nn.Linear(latent_dim, input_dim, bias=False)
    def forward(self, x):
        z = self.enc(x)
        return self.pred_dec(z), z

class LinearJEPA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.predictor = nn.Linear(latent_dim, latent_dim, bias=False)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data
    def forward(self, x_t, x_next):
        z_t = self.encoder(x_t)
        z_pred = self.predictor(z_t)
        with torch.no_grad(): z_target = self.target_encoder(x_next)
        return z_pred, z_target, z_t

class LinearProbabilisticVJEPA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.target_enc_mu = deepcopy(self.encoder)
        self.target_enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)
        self.pred_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_logvar = nn.Linear(latent_dim, latent_dim, bias=False)
        for p in self.target_enc_mu.parameters(): p.requires_grad = False
        for p in self.target_enc_logvar.parameters(): p.requires_grad = False

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_enc_mu.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def forward(self, x_t, x_next):
        z_t = self.encoder(x_t)
        p_mu = self.pred_mu(z_t)
        p_logvar = self.pred_logvar(z_t)
        with torch.no_grad():
            t_mu = self.target_enc_mu(x_next)
            t_logvar = self.target_enc_logvar(x_next)
        std = torch.exp(0.5 * t_logvar)
        z_target_sample = t_mu + torch.randn_like(std) * std
        return z_target_sample, (p_mu, p_logvar), (t_mu, t_logvar), z_t

# --- Bayesian JEPA (BJEPA) ---
class LinearBJEPA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.target_enc_mu = deepcopy(self.encoder)
        self.target_enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)
        for p in self.target_enc_mu.parameters(): p.requires_grad = False
        for p in self.target_enc_logvar.parameters(): p.requires_grad = False
        self.pred_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_logvar = nn.Linear(latent_dim, latent_dim, bias=False)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_enc_mu.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def product_of_experts(self, mu1, logvar1, mu2, logvar2):
        prec1 = torch.exp(-logvar1)
        prec2 = torch.exp(-logvar2)
        prec_post = prec1 + prec2
        var_post = 1.0 / prec_post
        mu_post = (mu1 * prec1 + mu2 * prec2) * var_post
        logvar_post = torch.log(var_post)
        return mu_post, logvar_post

    def forward(self, x_t, x_next):
        z_t = self.encoder(x_t)
        dyn_mu = self.pred_mu(z_t)
        dyn_logvar = self.pred_logvar(z_t)
        batch_size = x_t.size(0)
        prior_mu = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)
        post_mu, post_logvar = self.product_of_experts(dyn_mu, dyn_logvar, prior_mu, prior_logvar)
        with torch.no_grad():
            t_mu = self.target_enc_mu(x_next)
            t_logvar = self.target_enc_logvar(x_next)
        std = torch.exp(0.5 * t_logvar)
        z_target_sample = t_mu + torch.randn_like(std) * std
        return z_target_sample, (dyn_mu, dyn_logvar), (prior_mu, prior_logvar), (t_mu, t_logvar)

# ==========================================
# Loss Functions
# ==========================================
def vicreg_loss(x, y, sim=25.0, std=25.0, cov=1.0):
    repr_loss = nn.functional.mse_loss(x, y)
    std_loss = torch.mean(torch.relu(1 - torch.sqrt(x.var(0)+1e-4))) + \
               torch.mean(torch.relu(1 - torch.sqrt(y.var(0)+1e-4)))
    x = x - x.mean(0); y = y - y.mean(0)
    cov_loss = ((((x.T@x)/(x.size(0)-1))**2).sum() - ((x.T@x).diag()/(x.size(0)-1)**2).sum()) / x.size(1)
    return sim*repr_loss + std*std_loss + cov*cov_loss

def vjepa_prob_loss(z_sample, p_params, t_params, beta=0.01):
    p_mu, p_logvar = p_params
    t_mu, t_logvar = t_params
    p_var = torch.exp(p_logvar)
    nll = 0.5 * torch.mean(torch.sum(torch.log(p_var) + (z_sample - p_mu)**2 / p_var, dim=1))
    kl = -0.5 * torch.mean(torch.sum(1 + t_logvar - t_mu.pow(2) - t_logvar.exp(), dim=1))
    return nll + beta * kl

def bjepa_loss(z_sample, dyn_params, prior_params, t_params, beta=0.01, gamma=0.1):
    loss_vjepa = vjepa_prob_loss(z_sample, dyn_params, t_params, beta)
    d_mu, d_logvar = dyn_params
    pr_mu, pr_logvar = prior_params
    var_rat = torch.exp(d_logvar - pr_logvar)
    kl_prior = 0.5 * torch.mean(torch.sum(var_rat + (pr_mu - d_mu)**2 / torch.exp(pr_logvar) - 1 - (d_logvar - pr_logvar), dim=1))
    return loss_vjepa + gamma * kl_prior

# ==========================================
# 3. Visualization Helpers
# ==========================================
def calculate_snr(s, d_scaled):
    s_power = torch.mean(s ** 2)
    n_power = torch.mean(d_scaled ** 2)
    if n_power == 0: return float('inf')
    snr = 10 * torch.log10(s_power / n_power)
    return snr.item()


def visualize_dynamics_fixed(env, s_fixed, d_base, scale_values):
    """
    Visualizes the dynamics for 3 specific noise scales using a snippet of the data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Limit visualization to the first 100 steps
    # We slice input to 101 so observe() returns exactly 100 steps
    limit = 100
    s_in = s_fixed[:limit+1]
    d_in = d_base[:limit+1]

    for i, scale in enumerate(scale_values):
        # Generate exact observation for this scale from the fixed processes
        x_obs, _, s_out, d_out = env.observe(s_in, d_in, scale)
        snr = calculate_snr(s_out, d_out)
        ax = axes[i]
        ax.plot(s_out[:, 0].numpy(), label='Signal (s)', color='green', linewidth=3)
        ax.plot(d_out[:, 0].numpy(), label='Distractor (d)', color='red', alpha=0.5, linestyle='--')
        ax.plot(x_obs[:, 0].numpy(), label='Obs (x)', color='black', alpha=0.3)
        ax.set_title(f"Scale = {scale:.1f} (SNR: {snr:.1f} dB)", fontsize=12)
        ax.set_ylim(-10, 15)
        if i == 0: ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Dynamics: Signal (Green) vs Noise (Red) - Training Data ({limit} points)", fontsize=16)
    plt.tight_layout()
    plt.show()

# --- 3-Scale Row Plot ---
def plot_reconstructions_row(snapshots):
    """
    Plots 3 snapshots side-by-side in a single row.
    snapshots: list of dicts {'scale', 'snr', 'models', 'data':(x_tr, s_tr, x_te, s_te)}
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    colors = {'VAE': '#d62728', 'AR': '#ff7f0e', 'JEPA': '#17becf', 'VJEPA': '#1f77b4', 'BJEPA': '#9467bd'}
    styles = {'VAE': '--', 'AR': '--', 'JEPA': '-.', 'VJEPA': '-', 'BJEPA': '-'}
    t_steps = 100

    for i, snap in enumerate(snapshots):
        ax = axes[i]
        scale = snap['scale']
        snr = snap['snr']
        models = snap['models']
        # Unpack data
        x_train, s_train, x_test, s_test = snap['data']

        # Plot True Signal
        s_true = s_test[:t_steps, 0].numpy()
        ax.plot(s_true, label='True Signal (Test)', color='black', linewidth=3, alpha=0.6)

        with torch.no_grad():
            for name, model in models.items():
                # 1. Extract Latents
                if name == 'VAE':
                    z_tr = model.enc_mu(x_train).detach().numpy()
                    z_te = model.enc_mu(x_test).detach().numpy()
                elif name == 'AR':
                    z_tr = model.enc(x_train).detach().numpy()
                    z_te = model.enc(x_test).detach().numpy()
                elif name == 'JEPA':
                    z_tr = model.encoder(x_train).detach().numpy()
                    z_te = model.encoder(x_test).detach().numpy()
                elif name == 'VJEPA':
                    z_tr = model.encoder(x_train).detach().numpy()
                    z_te = model.encoder(x_test).detach().numpy()
                elif name == 'BJEPA':
                    z_tr = model.encoder(x_train).detach().numpy()
                    z_te = model.encoder(x_test).detach().numpy()

                # 2. FIT PROBE ON TRAINING DATA
                probe = LinearRegression().fit(z_tr, s_train.numpy())

                # 3. PREDICT ON TEST DATA
                pred = probe.predict(z_te)[:t_steps, 0]

                # Highlight BJEPA/VJEPA
                lw = 2.5 if name in ['BJEPA', 'VJEPA'] else 1.5
                alpha = 0.9 if name in ['BJEPA', 'VJEPA'] else 0.7
                ax.plot(pred, label=name, color=colors[name], linestyle=styles[name], linewidth=lw, alpha=alpha)

        ax.set_title(f"Scale {scale:.1f} (SNR: {snr:.1f} dB)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step")
        if i == 0:
            ax.set_ylabel("Signal Amplitude")
            ax.legend(loc='upper right', fontsize=10, ncol=2)

    plt.suptitle("Latent Reconstructions at 3 Noise Scales (Probe fit on Train, Applied to Test)", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Experiment Runner
# ==========================================
def run_experiment_fixed_signal(noise_scales, seed=seed):
    dim_s, dim_d, dim_x, dim_z = 4, 4, 20, 4
    models_list = ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']

    torch.manual_seed(seed); np.random.seed(seed)
    env = DistractorSystem(dim_s, dim_d, dim_x)

    print("Generating Fixed Signal and Base Distractor...")
    s_train_fixed, d_train_base = env.generate_latent_processes(6000)
    s_test_fixed, d_test_base = env.generate_latent_processes(2000)

    # target_indices = [0, len(noise_scales)//2, len(noise_scales)-1]

    # Extract the actual scale values for visualization
    vis_scales = [noise_scales[i] for i in target_indices]

    # Feed Training Data and Specific Scales
    visualize_dynamics_fixed(env, s_train_fixed, d_train_base, vis_scales)

    results = {
        'signal_train': {m: [] for m in models_list}, 'signal_test': {m: [] for m in models_list},
        'distractor_train': {m: [] for m in models_list}, 'distractor_test': {m: [] for m in models_list}
    }

    print(f"\nRunning Models (Seed {seed})...")
    print(f"{'Scale (SNR)':<20} | {'Model':<6} | {'Signal R2 (Tr/Te)':<19} | {'Noise R2 (Tr/Te)':<18} | {'Time (Tr/Te)':<14}")
    print("-" * 85)

    # Indices to capture for row plotting: Start, Middle, End
    # target_indices = [0, len(noise_scales)//2, len(noise_scales)-1]
    snapshots = []
    final_snr = 0.0

    for i, ns in enumerate(noise_scales):
        x_train, x_train_next, s_train, d_train = env.observe(s_train_fixed, d_train_base, ns)
        x_test, x_test_next, s_test, d_test = env.observe(s_test_fixed, d_test_base, ns)

        snr = calculate_snr(s_train, d_train)
        snr_str = f"{ns:.1f} ({snr:.1f} dB)"
        if ns == noise_scales[-1]: final_snr = snr

        xm, xs = x_train.mean(0), x_train.std(0)
        x_train = (x_train - xm)/(xs+1e-6); x_train_next = (x_train_next - xm)/(xs+1e-6)
        x_test = (x_test - xm)/(xs+1e-6); x_test_next = (x_test_next - xm)/(xs+1e-6)

        def eval_reps(z_tr, z_te, s_tr, s_te):
            probe_s = LinearRegression().fit(z_tr.detach().numpy(), s_tr.numpy())
            r2_tr = r2_score(s_tr.numpy(), probe_s.predict(z_tr.detach().numpy()))
            r2_te = r2_score(s_te.numpy(), probe_s.predict(z_te.detach().numpy()))
            return r2_tr, r2_te

        def train_eval_cycle(model, name, x_tr, x_tr_next, x_te, s_tr, s_te, d_tr, d_te):
            t0 = time.time()
            opt = optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(6000):
                opt.zero_grad()
                if name == 'VAE':
                    xh, mu, lv, _ = model(x_tr)
                    loss = nn.functional.mse_loss(xh, x_tr, reduction='sum') - 0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())
                elif name == 'AR':
                    loss = nn.functional.mse_loss(model(x_tr)[0], x_tr_next)
                elif name == 'JEPA':
                    zp, zt, _ = model(x_tr, x_tr_next)
                    loss = vicreg_loss(zp, zt)
                    model.update_target()
                elif name == 'VJEPA':
                    z_s, pp, tp, _ = model(x_tr, x_tr_next)
                    loss = vjepa_prob_loss(z_s, pp, tp)
                    model.update_target()
                elif name == 'BJEPA':
                    z_s, dyn_p, prior_p, target_p = model(x_tr, x_tr_next)
                    loss = bjepa_loss(z_s, dyn_p, prior_p, target_p)
                    model.update_target()
                loss.backward(); opt.step()
            train_time = time.time() - t0

            t0 = time.time()
            with torch.no_grad():
                if name == 'VAE': z_tr = model.enc_mu(x_tr); z_te = model.enc_mu(x_te)
                elif name == 'AR': z_tr = model.enc(x_tr); z_te = model.enc(x_te)
                elif name == 'JEPA': z_tr = model.encoder(x_tr); z_te = model.encoder(x_te)
                elif name == 'VJEPA': z_tr = model.encoder(x_tr); z_te = model.encoder(x_te)
                elif name == 'BJEPA': z_tr = model.encoder(x_tr); z_te = model.encoder(x_te)
            test_time = time.time() - t0

            sig_tr, sig_te = eval_reps(z_tr, z_te, s_tr, s_te)
            noi_tr, noi_te = eval_reps(z_tr, z_te, d_tr, d_te)
            return sig_tr, sig_te, noi_tr, noi_te, train_time, test_time

        # Run Models & Store Current Models
        current_models = {}

        # VAE
        vae = LinearVAE(dim_x, dim_z)
        st, se, nt, ne, tt, te = train_eval_cycle(vae, 'VAE', x_train, x_train_next, x_test, s_train, s_test, d_train, d_test)
        results['signal_train']['VAE'].append(st); results['signal_test']['VAE'].append(se)
        results['distractor_train']['VAE'].append(nt); results['distractor_test']['VAE'].append(ne)
        print(f"{snr_str:<20} | {'VAE':<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {tt:.1f}s / {te:.2f}s")
        current_models['VAE'] = vae

        # AR
        ar = LinearAR(dim_x, dim_z)
        st, se, nt, ne, tt, te = train_eval_cycle(ar, 'AR', x_train, x_train_next, x_test, s_train, s_test, d_train, d_test)
        results['signal_train']['AR'].append(st); results['signal_test']['AR'].append(se)
        results['distractor_train']['AR'].append(nt); results['distractor_test']['AR'].append(ne)
        print(f"{snr_str:<20} | {'AR':<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {tt:.1f}s / {te:.2f}s")
        current_models['AR'] = ar

        # JEPA
        jepa = LinearJEPA(dim_x, dim_z)
        st, se, nt, ne, tt, te = train_eval_cycle(jepa, 'JEPA', x_train, x_train_next, x_test, s_train, s_test, d_train, d_test)
        results['signal_train']['JEPA'].append(st); results['signal_test']['JEPA'].append(se)
        results['distractor_train']['JEPA'].append(nt); results['distractor_test']['JEPA'].append(ne)
        print(f"{snr_str:<20} | {'JEPA':<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {tt:.1f}s / {te:.2f}s")
        current_models['JEPA'] = jepa

        # VJEPA
        vjepa = LinearProbabilisticVJEPA(dim_x, dim_z)
        st, se, nt, ne, tt, te = train_eval_cycle(vjepa, 'VJEPA', x_train, x_train_next, x_test, s_train, s_test, d_train, d_test)
        results['signal_train']['VJEPA'].append(st); results['signal_test']['VJEPA'].append(se)
        results['distractor_train']['VJEPA'].append(nt); results['distractor_test']['VJEPA'].append(ne)
        print(f"{snr_str:<20} | {'VJEPA':<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {tt:.1f}s / {te:.2f}s")
        current_models['VJEPA'] = vjepa

        # BJEPA
        bjepa = LinearBJEPA(dim_x, dim_z)
        st, se, nt, ne, tt, te = train_eval_cycle(bjepa, 'BJEPA', x_train, x_train_next, x_test, s_train, s_test, d_train, d_test)
        results['signal_train']['BJEPA'].append(st); results['signal_test']['BJEPA'].append(se)
        results['distractor_train']['BJEPA'].append(nt); results['distractor_test']['BJEPA'].append(ne)
        print(f"{snr_str:<20} | {'BJEPA':<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {tt:.1f}s / {te:.2f}s")
        print("-" * 85)
        current_models['BJEPA'] = bjepa

        # --- SNAPSHOT LOGIC ---
        if i in target_indices:
            snapshots.append({
                'scale': ns,
                'snr': snr,
                'models': deepcopy(current_models),
                'data': (x_train, s_train, x_test, s_test)
            })

    # Return snapshots for external usage if needed
    artifacts = {
        'last_models': current_models, # models from final iteration
        'data': {
            'x_train': x_train, 's_train': s_train,
            'x_test': x_test,   's_test': s_test
        },
        'final_snr': final_snr,
        'snapshots': snapshots # The 3 captured states
    }

    # Plot the 3-scale reconstruction row
    plot_reconstructions_row(snapshots)

    return results, noise_scales, artifacts

# Run
scales = np.linspace(0, 8.0, 9)
target_indices = [0, len(scales)//2, len(scales)-1]
results, scales, artifacts = run_experiment_fixed_signal(scales, seed=seed)





# Plotting 2x2 Grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = {'VAE': '#d62728', 'AR': '#ff7f0e', 'JEPA': '#17becf', 'VJEPA': '#1f77b4', 'BJEPA': '#9467bd'}
markers = {'VAE': 'o', 'AR': 'x', 'JEPA': 'v', 'VJEPA': 's', 'BJEPA': 'D'}

# 1. Training Signal Recovery
ax = axes[0, 0]
for k in ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']:
    ax.plot(scales, results['signal_train'][k], label=k, color=colors[k], linestyle='--', marker=markers[k], linewidth=2, alpha=0.7)
ax.set_ylabel(r'Train $R^2$', fontsize=16); ax.set_title(r'Signal Recovery (Training)', fontsize=18)
ax.set_xlabel('Distractor Noise Scale', fontsize=16); ax.grid(True, alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=16)

# 2. Training Distractor Recovery
ax = axes[0, 1]
for k in ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']:
    ax.plot(scales[1:], results['distractor_train'][k][1:], label=k, color=colors[k], linestyle='--', marker=markers[k], linewidth=2, alpha=0.7)
ax.set_ylabel(r'Train $R^2$', fontsize=16); ax.set_title(r'Distractor Recovery (Training)', fontsize=18)
ax.set_xlabel('Distractor Noise Scale', fontsize=16); ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=16); ax.tick_params(axis='both', which='major', labelsize=16)

# 3. Test Signal Recovery
ax = axes[1, 0]
for k in ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']:
    ax.plot(scales, results['signal_test'][k], label=k, color=colors[k], linestyle='-', marker=markers[k], linewidth=2)
ax.set_ylabel(r'Test $R^2$', fontsize=16); ax.set_title(r'Signal Recovery (Test) - Generalization', fontsize=18)
ax.set_xlabel('Distractor Noise Scale', fontsize=16); ax.grid(True, alpha=0.3); ax.legend(loc='lower left', fontsize=16); ax.tick_params(axis='both', which='major', labelsize=16)

# 4. Test Distractor Recovery
ax = axes[1, 1]
for k in ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']:
    ax.plot(scales[1:], results['distractor_test'][k][1:], label=k, color=colors[k], linestyle='-', marker=markers[k], linewidth=2)
ax.set_ylabel(r'Test $R^2$', fontsize=16); ax.set_title(r'Distractor Recovery (Test)', fontsize=18)
ax.set_xlabel('Distractor Noise Scale', fontsize=16); ax.grid(True, alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout(); plt.show()
















"""# save results."""

import os
import torch
import pickle
import numpy as np
import json

# ==========================================
# Configuration
# ==========================================
OUTPUT_DIR = "vjepa_and_bjepa_experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving experiment artifacts to: {os.path.abspath(OUTPUT_DIR)}")

# ==========================================
# 1. Save Metrics & Metadata (Pickle)
# ==========================================
metadata = {
    'seed': seed,
    'noise_scales': scales.tolist() if isinstance(scales, np.ndarray) else scales,
    'final_snr': artifacts['final_snr'],
    'results': results
}

pkl_path = os.path.join(OUTPUT_DIR, "metrics_and_meta.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"[x] Metrics and Metadata saved to {pkl_path}")

# ==========================================
# 2. Save Raw Datasets (Final Run)
# ==========================================
data_path = os.path.join(OUTPUT_DIR, "datasets_final.npz")
np.savez_compressed(
    data_path,
    x_train=artifacts['data']['x_train'].cpu().numpy(),
    s_train=artifacts['data']['s_train'].cpu().numpy(),
    x_test=artifacts['data']['x_test'].cpu().numpy(),
    s_test=artifacts['data']['s_test'].cpu().numpy()
)
print(f"[x] Final Datasets saved to {data_path}")

# ==========================================
# 3. Save Final Models
# ==========================================
model_weights = {}
for name, model in artifacts['last_models'].items():
    model_weights[name] = model.state_dict()

weights_path = os.path.join(OUTPUT_DIR, "model_weights_final.pth")
torch.save(model_weights, weights_path)
print(f"[x] Final Model weights saved to {weights_path}")

# ==========================================
# 4. NEW: Save Snapshots (For Row Plot)
# ==========================================
# We need to extract state_dicts and numpy arrays from the snapshots list
# so they can be saved efficiently without pickling entire model objects.
serializable_snapshots = []

for snap in artifacts['snapshots']:
    # 1. Extract weights
    models_state = {name: model.state_dict() for name, model in snap['models'].items()}

    # 2. Extract data (convert to numpy to save space/detach)
    x_tr, s_tr, x_te, s_te = snap['data']
    data_numpy = {
        'x_tr': x_tr.cpu().numpy(), 's_tr': s_tr.cpu().numpy(),
        'x_te': x_te.cpu().numpy(), 's_te': s_te.cpu().numpy()
    }

    serializable_snapshots.append({
        'scale': snap['scale'],
        'snr': snap['snr'],
        'models': models_state,
        'data': data_numpy
    })

snapshot_path = os.path.join(OUTPUT_DIR, "snapshots.pth")
torch.save(serializable_snapshots, snapshot_path)
print(f"[x] Snapshots (Low/Mid/High noise models) saved to {snapshot_path}")

# ==========================================
# 5. Save the Figures
# ==========================================
# Note: In a notebook, 'fig' refers to the last active figure.
# If you generated two figures (the 2x2 grid and the row plot),
# you might want to save them manually if they aren't in the 'fig' variable.
try:
    # Attempt to save the currently active figure
    plot_path = os.path.join(OUTPUT_DIR, "latest_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[x] Active plot saved to {plot_path}")
except Exception as e:
    print(f"[!] Could not save plot automatically: {e}")

print(f"\nSUCCESS: All experiment data saved to '{OUTPUT_DIR}'")

