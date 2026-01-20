# -*- coding: utf-8 -*-

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""
[20/01/2026] Corrected Toy Experiment Script

Key corrections from the original:
1. JEPA/VJEPA/BJEPA use predictor output (predicted Z_{t+1}) for linear probe evaluation
2. BJEPA uses hard fusion (Product of Experts) at inference time
3. Proper temporal alignment: predictive models evaluated against s_{t+1}, VAE against s_t
4. AR's latent z_t is designed to predict x_{t+1}, so evaluated against s_{t+1}

Based on our paper:
- VAE: Static encoder, z_t represents current state → evaluate against s_t
- AR: Predictive decoder in pixel space, z_t predicts x_{t+1} → evaluate against s_{t+1}
- JEPA: Deterministic predictor, z_pred = predictor(encoder(x_t)) predicts Z_{t+1} → evaluate against s_{t+1}
- VJEPA: Probabilistic predictor, p_mu = pred_mu(encoder(x_t)) predicts E[Z_{t+1}] → evaluate against s_{t+1}
- BJEPA: At inference, use hard fusion (PoE) of dynamics + prior → evaluate against s_{t+1}
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
seed = 111
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
        """Generate the underlying signal and distractor processes."""
        s = torch.zeros(n_samples + 1, self.dim_s)
        s[0] = torch.randn(self.dim_s)
        d_base = torch.zeros(n_samples + 1, self.dim_d)
        d_base[0] = torch.randn(self.dim_d)

        for t in range(n_samples):
            # Signal: stable rotation
            s[t+1] = s[t] @ self.A.T + torch.randn(self.dim_s) * self.Q_s
            # Distractor: sticky random walk
            d_base[t+1] = d_base[t] @ self.A_distractor.T + torch.randn(self.dim_d) * 0.3

        return s, d_base

    def observe(self, s, d_base, noise_scale):
        """
        Generate observations from latent processes.

        Returns:
            x_t: observations at time t (indices 0 to n_samples-1)
            x_next: observations at time t+1 (indices 1 to n_samples)
            s_t: signal at time t
            d_t: scaled distractor at time t
        """
        n_samples = s.shape[0] - 1
        d_scaled = d_base * noise_scale
        sensor_noise = torch.randn(n_samples + 1, self.dim_x) * 0.01
        x = s @ self.C.T + d_scaled @ self.D.T + sensor_noise
        # x[:-1] is x_t, x[1:] is x_{t+1}
        # s[:-1] is s_t, s[1:] is s_{t+1}
        return x[:-1], x[1:], s[:-1], d_scaled[:-1]


# ==========================================
# 2. Models
# ==========================================

class LinearVAE(nn.Module):
    """
    Variational Autoencoder - static model.
    Encodes x_t to z_t representing the CURRENT state.
    """
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

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For VAE: return encoder output representing CURRENT state.
        This should be evaluated against s_t.
        """
        return self.enc_mu(x_t)


class LinearAR(nn.Module):
    """
    Autoregressive model - predictive in pixel space.
    z_t = enc(x_t) is used to predict x_{t+1}.
    Therefore z_t should contain information about s_{t+1}.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, latent_dim, bias=False)
        self.pred_dec = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.enc(x)
        return self.pred_dec(z), z

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For AR: return encoder output.
        Since AR predicts x_{t+1} from z_t, z_t should be evaluated against s_{t+1}.
        """
        return self.enc(x_t)


class LinearJEPA(nn.Module):
    """
    Deterministic JEPA.
    Predictor output z_pred = predictor(encoder(x_t)) predicts Z_{t+1}.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.predictor = nn.Linear(latent_dim, latent_dim, bias=False)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def forward(self, x_t, x_next):
        """Training forward pass."""
        z_t = self.encoder(x_t)
        z_pred = self.predictor(z_t)  # Prediction of Z_{t+1}
        with torch.no_grad():
            z_target = self.target_encoder(x_next)
        return z_pred, z_target, z_t

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For JEPA: return PREDICTOR output (predicted Z_{t+1}).
        This should be evaluated against s_{t+1}.
        """
        z_t = self.encoder(x_t)
        z_pred = self.predictor(z_t)
        return z_pred


class LinearProbabilisticVJEPA(nn.Module):
    """
    Probabilistic VJEPA.
    Predictor outputs distribution parameters for Z_{t+1}.
    At inference, use the predicted mean as point estimate.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.target_enc_mu = deepcopy(self.encoder)
        self.target_enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)
        self.pred_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_logvar = nn.Linear(latent_dim, latent_dim, bias=False)

        for p in self.target_enc_mu.parameters():
            p.requires_grad = False
        for p in self.target_enc_logvar.parameters():
            p.requires_grad = False

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_enc_mu.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def forward(self, x_t, x_next):
        """Training forward pass."""
        z_t = self.encoder(x_t)
        p_mu = self.pred_mu(z_t)  # Predicted mean of Z_{t+1}
        p_logvar = self.pred_logvar(z_t)

        with torch.no_grad():
            t_mu = self.target_enc_mu(x_next)
            t_logvar = self.target_enc_logvar(x_next)

        # Sample from target distribution for training
        std = torch.exp(0.5 * t_logvar)
        z_target_sample = t_mu + torch.randn_like(std) * std

        return z_target_sample, (p_mu, p_logvar), (t_mu, t_logvar), z_t

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For VJEPA: return predicted MEAN of Z_{t+1}.
        This should be evaluated against s_{t+1}.
        """
        z_t = self.encoder(x_t)
        p_mu = self.pred_mu(z_t)
        return p_mu


class LinearBJEPA(nn.Module):
    """
    Bayesian JEPA with Product of Experts.

    Training: Soft fusion via KL regularization (dynamics learns independently)
    Inference: Hard fusion via PoE (combines dynamics + prior)

    From the paper (Section 6.3):
    - Training phase uses soft fusion where prior acts as regularizer
    - Inference phase uses hard fusion (PoE) to intersect dynamics and prior manifolds
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.target_enc_mu = deepcopy(self.encoder)
        self.target_enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)

        for p in self.target_enc_mu.parameters():
            p.requires_grad = False
        for p in self.target_enc_logvar.parameters():
            p.requires_grad = False

        # Dynamics predictor (likelihood expert)
        self.pred_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_logvar = nn.Linear(latent_dim, latent_dim, bias=False)

        # Learnable static prior (constraint expert)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_enc_mu.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def product_of_experts(self, mu1, logvar1, mu2, logvar2):
        """
        Combine two Gaussian experts via Product of Experts.
        Returns the posterior mean and logvar.

        posterior_precision = precision_1 + precision_2
        posterior_mean = (precision_1 * mu1 + precision_2 * mu2) / posterior_precision
        """
        prec1 = torch.exp(-logvar1)
        prec2 = torch.exp(-logvar2)
        prec_post = prec1 + prec2
        var_post = 1.0 / prec_post
        mu_post = (mu1 * prec1 + mu2 * prec2) * var_post
        logvar_post = torch.log(var_post)
        return mu_post, logvar_post

    def forward(self, x_t, x_next):
        """
        Training forward pass.
        Returns dynamics prediction (for soft fusion training) and target.
        """
        z_t = self.encoder(x_t)

        # Dynamics expert prediction
        dyn_mu = self.pred_mu(z_t)
        dyn_logvar = self.pred_logvar(z_t)

        # Expand prior for batch
        batch_size = x_t.size(0)
        prior_mu = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)

        # Target distribution
        with torch.no_grad():
            t_mu = self.target_enc_mu(x_next)
            t_logvar = self.target_enc_logvar(x_next)

        # Sample from target for training loss
        std = torch.exp(0.5 * t_logvar)
        z_target_sample = t_mu + torch.randn_like(std) * std

        return z_target_sample, (dyn_mu, dyn_logvar), (prior_mu, prior_logvar), (t_mu, t_logvar)

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For BJEPA at inference: use HARD FUSION via Product of Experts.

        This combines:
        - Dynamics expert: p_like(Z_{t+1} | Z_t) - what the model predicts will happen
        - Prior expert: p_prior(Z_{t+1}) - structural constraints

        Returns the fused posterior mean, which should be evaluated against s_{t+1}.
        """
        z_t = self.encoder(x_t)

        # Dynamics expert
        dyn_mu = self.pred_mu(z_t)
        dyn_logvar = self.pred_logvar(z_t)

        # Expand prior for batch
        batch_size = x_t.size(0)
        prior_mu = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)

        # HARD FUSION: Product of Experts
        post_mu, post_logvar = self.product_of_experts(dyn_mu, dyn_logvar, prior_mu, prior_logvar)

        return post_mu


# ==========================================
# 3. Loss Functions
# ==========================================

def vicreg_loss(x, y, sim=25.0, std=25.0, cov=1.0):
    """VICReg loss for deterministic JEPA."""
    repr_loss = nn.functional.mse_loss(x, y)
    std_loss = torch.mean(torch.relu(1 - torch.sqrt(x.var(0) + 1e-4))) + \
               torch.mean(torch.relu(1 - torch.sqrt(y.var(0) + 1e-4)))
    x = x - x.mean(0)
    y = y - y.mean(0)
    cov_loss = ((((x.T @ x) / (x.size(0) - 1)) ** 2).sum() -
                ((x.T @ x).diag() / (x.size(0) - 1) ** 2).sum()) / x.size(1)
    return sim * repr_loss + std * std_loss + cov * cov_loss


def vjepa_prob_loss(z_sample, p_params, t_params, beta=0.01):
    """
    VJEPA probabilistic loss.
    - NLL: how well does predictor distribution explain the target sample
    - KL: regularize target distribution toward prior
    """
    p_mu, p_logvar = p_params
    t_mu, t_logvar = t_params
    p_var = torch.exp(p_logvar)

    # Negative log-likelihood of z_sample under predictor distribution
    nll = 0.5 * torch.mean(torch.sum(torch.log(p_var) + (z_sample - p_mu) ** 2 / p_var, dim=1))

    # KL divergence: regularize target encoder toward N(0,I)
    kl = -0.5 * torch.mean(torch.sum(1 + t_logvar - t_mu.pow(2) - t_logvar.exp(), dim=1))

    return nll + beta * kl


def bjepa_loss(z_sample, dyn_params, prior_params, t_params, beta=0.01, gamma=0.1):
    """
    BJEPA training loss with SOFT FUSION.

    From paper Eq. 16:
    L_BJEPA = L_VJEPA (dynamics fitting) + γ * KL(p_like || p_prior) (structural regularization)

    This is soft fusion: dynamics learns to fit the data while being regularized toward prior.
    """
    # Standard VJEPA loss for dynamics
    loss_vjepa = vjepa_prob_loss(z_sample, dyn_params, t_params, beta)

    # KL between dynamics and prior (soft fusion regularization)
    d_mu, d_logvar = dyn_params
    pr_mu, pr_logvar = prior_params

    var_rat = torch.exp(d_logvar - pr_logvar)
    kl_prior = 0.5 * torch.mean(torch.sum(
        var_rat + (pr_mu - d_mu) ** 2 / torch.exp(pr_logvar) - 1 - (d_logvar - pr_logvar),
        dim=1
    ))

    return loss_vjepa + gamma * kl_prior


# ==========================================
# 4. Visualization Helpers
# ==========================================

def calculate_snr(s, d_scaled):
    """Calculate Signal-to-Noise Ratio in dB."""
    s_power = torch.mean(s ** 2)
    n_power = torch.mean(d_scaled ** 2)
    if n_power == 0:
        return float('inf')
    snr = 10 * torch.log10(s_power / n_power)
    return snr.item()


def visualize_dynamics_fixed(env, s_fixed, d_base, scale_values, snr_values=None):
    """
    Visualize dynamics at 3 specific noise scales.

    Args:
        env: DistractorSystem environment
        s_fixed: Full signal trajectory
        d_base: Full distractor base trajectory
        scale_values: List of 3 noise scales to visualize
        snr_values: Pre-computed SNR values from full training data (optional).
                    If None, computes SNR from visualization subset (less accurate).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    limit = 100
    s_in = s_fixed[:limit + 1]
    d_in = d_base[:limit + 1]

    for i, scale in enumerate(scale_values):
        x_obs, _, s_out, d_out = env.observe(s_in, d_in, scale)

        # Use pre-computed SNR from full data if available, otherwise compute from subset
        if snr_values is not None:
            snr = snr_values[i]
        else:
            snr = calculate_snr(s_out, d_out)

        ax = axes[i]
        ax.plot(s_out[:, 0].numpy(), label='Signal (s)', color='green', linewidth=3)
        ax.plot(d_out[:, 0].numpy(), label='Distractor (d)', color='red', alpha=0.5, linestyle='--')
        ax.plot(x_obs[:, 0].numpy(), label='Obs (x)', color='black', alpha=0.3)
        ax.set_title(f"Scale = {scale:.1f} (SNR: {snr:.1f} dB)", fontsize=12)
        ax.set_ylim(-10, 15)
        if i == 0:
            ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Dynamics: Signal (Green) vs Noise (Red) - Training Data ({limit} points)", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_reconstructions_row(snapshots):
    """
    Plot latent reconstructions at 3 noise scales.

    Key: Shows how well each model's latent (via linear probe) recovers the TRUE signal.
    - VAE: evaluated against s_t (current signal)
    - AR/JEPA/VJEPA/BJEPA: evaluated against s_{t+1} (next signal)
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
        data = snap['data']

        # Unpack data
        x_tr, x_tr_next, s_tr, s_tr_next, d_tr, d_tr_next = data['train']
        x_te, x_te_next, s_te, s_te_next, d_te, d_te_next = data['test']

        # For visualization, we show predictions against the NEXT signal (s_{t+1})
        # since most models are predictive
        s_true = s_te_next[:t_steps, 0].numpy()
        ax.plot(s_true, label='True Signal $s_{t+1}$ (Test)', color='black', linewidth=3, alpha=0.6)

        with torch.no_grad():
            for name, model in models.items():
                # Get appropriate latent representation
                if name == 'VAE':
                    # VAE represents current state - fit probe on s_t, but show prediction
                    # for fair comparison, we'll still show what it predicts for s_{t+1}
                    z_tr = model.get_latent_for_probe(x_tr).detach().numpy()
                    z_te = model.get_latent_for_probe(x_te).detach().numpy()
                    # Note: VAE's z_t is fit to s_t, so predicting s_{t+1} will be worse
                    # This is expected behavior showing VAE doesn't predict futures
                    probe = LinearRegression().fit(z_tr, s_tr_next.numpy())
                else:
                    # AR, JEPA, VJEPA, BJEPA - all predictive models
                    z_tr = model.get_latent_for_probe(x_tr, x_tr_next).detach().numpy()
                    z_te = model.get_latent_for_probe(x_te, x_te_next).detach().numpy()
                    probe = LinearRegression().fit(z_tr, s_tr_next.numpy())

                pred = probe.predict(z_te)[:t_steps, 0]

                lw = 2.5 if name in ['BJEPA', 'VJEPA'] else 1.5
                alpha = 0.9 if name in ['BJEPA', 'VJEPA'] else 0.7
                ax.plot(pred, label=f'{name}', color=colors[name],
                        linestyle=styles[name], linewidth=lw, alpha=alpha)

        ax.set_title(f"Scale {scale:.1f} (SNR: {snr:.1f} dB)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step (t)")
        if i == 0:
            ax.set_ylabel("Signal Amplitude")
            ax.legend(loc='upper right', fontsize=10, ncol=2)

    plt.suptitle("Latent Reconstructions: Predicting $s_{t+1}$ from Latent Representations", fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. Experiment Runner
# ==========================================

def run_experiment_fixed_signal(noise_scales, target_indices, seed=seed):
    """
    Run the full experiment with correct temporal alignment.

    Key corrections:
    1. VAE: z_t evaluated against s_t (current state)
    2. AR/JEPA/VJEPA/BJEPA: predictive latent evaluated against s_{t+1}
    3. BJEPA uses hard fusion (PoE) at inference time
    """
    dim_s, dim_d, dim_x, dim_z = 4, 4, 20, 4
    models_list = ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = DistractorSystem(dim_s, dim_d, dim_x)

    print("Generating Fixed Signal and Base Distractor...")
    s_train_fixed, d_train_base = env.generate_latent_processes(6000)
    s_test_fixed, d_test_base = env.generate_latent_processes(2000)

    # Visualization scales
    vis_scales = [noise_scales[i] for i in target_indices]

    # Pre-compute SNR values from FULL training data for visualization
    # This ensures consistency with the SNR values reported in the results table
    vis_snr_values = []
    for scale in vis_scales:
        _, _, s_full, d_full = env.observe(s_train_fixed, d_train_base, scale)
        snr_full = calculate_snr(s_full, d_full)
        vis_snr_values.append(snr_full)

    visualize_dynamics_fixed(env, s_train_fixed, d_train_base, vis_scales, snr_values=vis_snr_values)

    results = {
        'signal_train': {m: [] for m in models_list},
        'signal_test': {m: [] for m in models_list},
        'distractor_train': {m: [] for m in models_list},
        'distractor_test': {m: [] for m in models_list}
    }

    print(f"\nRunning Models (Seed {seed})...")
    print(f"{'Scale (SNR)':<20} | {'Model':<6} | {'Signal R2 (Tr/Te)':<19} | {'Noise R2 (Tr/Te)':<18} | {'Time (Tr/Te)':<14}")
    print("-" * 85)

    snapshots = []
    final_snr = 0.0

    for i, ns in enumerate(noise_scales):
        # Generate observations
        x_train, x_train_next, s_train, d_train = env.observe(s_train_fixed, d_train_base, ns)
        x_test, x_test_next, s_test, d_test = env.observe(s_test_fixed, d_test_base, ns)

        # Get the NEXT time step signals for predictive models
        # observe() returns:
        #   x_train = x[:-1] (indices 0 to n-1), shape: n_samples
        #   x_train_next = x[1:] (indices 1 to n), shape: n_samples
        #   s_train = s[:-1] (indices 0 to n-1), shape: n_samples
        #
        # For predictive models evaluating against s_{t+1}:
        #   When we have x_t, we predict z_{t+1}, which should match s_{t+1}
        #   s_train corresponds to s_t (same indexing as x_train)
        #   s_train_next should be s_{t+1}, which is s_train_fixed[1:n] where n = len(x_train)
        #
        # s_train_fixed has shape (n_samples+1,) = (6001,)
        # s_train = s_train_fixed[:-1] has shape (6000,) - this is s_t
        # s_train_next = s_train_fixed[1:] has shape (6000,) - this is s_{t+1}
        # But we need to match x_train which has shape (6000,)
        # So s_train_next = s_train_fixed[1:-1+1] = s_train_fixed[1:] but trimmed to match

        # Actually, let's trace through carefully:
        # s_train_fixed: indices 0, 1, 2, ..., 6000 (6001 elements)
        # observe() does: s[:-1] -> indices 0, 1, ..., 5999 (6000 elements) = s_train
        # x_train[i] corresponds to s_train[i] = s_train_fixed[i]
        # x_train_next[i] corresponds to s_train_fixed[i+1]
        # So s_train_next[i] should equal s_train_fixed[i+1] for i in 0..5999
        # That means s_train_next = s_train_fixed[1:6001] = s_train_fixed[1:][:6000]

        n_train = x_train.shape[0]
        n_test = x_test.shape[0]

        s_train_next = s_train_fixed[1:1+n_train]  # s_{t+1} aligned with x_train
        s_test_next = s_test_fixed[1:1+n_test]

        d_train_next = d_train_base[1:1+n_train] * ns
        d_test_next = d_test_base[1:1+n_test] * ns

        snr = calculate_snr(s_train, d_train)
        snr_str = f"{ns:.1f} ({snr:.1f} dB)"
        if ns == noise_scales[-1]:
            final_snr = snr

        # Normalize observations
        xm, xs = x_train.mean(0), x_train.std(0)
        x_train_norm = (x_train - xm) / (xs + 1e-6)
        x_train_next_norm = (x_train_next - xm) / (xs + 1e-6)
        x_test_norm = (x_test - xm) / (xs + 1e-6)
        x_test_next_norm = (x_test_next - xm) / (xs + 1e-6)

        def eval_reps_with_alignment(model, name, x_tr, x_tr_next, x_te, x_te_next,
                                      s_tr, s_te, s_tr_next, s_te_next,
                                      d_tr, d_te, d_tr_next, d_te_next):
            """
            Evaluate representations with correct temporal alignment.

            VAE: z_t should predict s_t (current)
            AR/JEPA/VJEPA/BJEPA: predictive latent should predict s_{t+1} (next)
            """
            with torch.no_grad():
                if name == 'VAE':
                    # VAE encodes current state
                    z_tr = model.get_latent_for_probe(x_tr)
                    z_te = model.get_latent_for_probe(x_te)
                    # Evaluate against CURRENT signal
                    signal_tr, signal_te = s_tr, s_te
                    distractor_tr, distractor_te = d_tr, d_te
                else:
                    # Predictive models
                    z_tr = model.get_latent_for_probe(x_tr, x_tr_next)
                    z_te = model.get_latent_for_probe(x_te, x_te_next)
                    # Evaluate against NEXT signal
                    signal_tr, signal_te = s_tr_next, s_te_next
                    distractor_tr, distractor_te = d_tr_next, d_te_next

            # Fit linear probe and evaluate
            z_tr_np = z_tr.detach().numpy()
            z_te_np = z_te.detach().numpy()

            # Signal R^2
            probe_s = LinearRegression().fit(z_tr_np, signal_tr.numpy())
            r2_sig_tr = r2_score(signal_tr.numpy(), probe_s.predict(z_tr_np))
            r2_sig_te = r2_score(signal_te.numpy(), probe_s.predict(z_te_np))

            # Distractor R^2
            probe_d = LinearRegression().fit(z_tr_np, distractor_tr.numpy())
            r2_dist_tr = r2_score(distractor_tr.numpy(), probe_d.predict(z_tr_np))
            r2_dist_te = r2_score(distractor_te.numpy(), probe_d.predict(z_te_np))

            return r2_sig_tr, r2_sig_te, r2_dist_tr, r2_dist_te

        def train_model(model, name, x_tr, x_tr_next, num_epochs=6000):
            """Train a single model."""
            opt = optim.Adam(model.parameters(), lr=1e-3)

            for _ in range(num_epochs):
                opt.zero_grad()

                if name == 'VAE':
                    xh, mu, lv, _ = model(x_tr)
                    loss = nn.functional.mse_loss(xh, x_tr, reduction='sum') - \
                           0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())

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

                loss.backward()
                opt.step()

        # Train and evaluate all models
        current_models = {}

        for name in models_list:
            t0 = time.time()

            # Initialize model
            if name == 'VAE':
                model = LinearVAE(dim_x, dim_z)
            elif name == 'AR':
                model = LinearAR(dim_x, dim_z)
            elif name == 'JEPA':
                model = LinearJEPA(dim_x, dim_z)
            elif name == 'VJEPA':
                model = LinearProbabilisticVJEPA(dim_x, dim_z)
            elif name == 'BJEPA':
                model = LinearBJEPA(dim_x, dim_z)

            # Train
            train_model(model, name, x_train_norm, x_train_next_norm)
            train_time = time.time() - t0

            # Evaluate
            t0 = time.time()
            st, se, nt, ne = eval_reps_with_alignment(
                model, name,
                x_train_norm, x_train_next_norm, x_test_norm, x_test_next_norm,
                s_train, s_test, s_train_next, s_test_next,
                d_train, d_test, d_train_next, d_test_next
            )
            test_time = time.time() - t0

            # Store results
            results['signal_train'][name].append(st)
            results['signal_test'][name].append(se)
            results['distractor_train'][name].append(nt)
            results['distractor_test'][name].append(ne)

            print(f"{snr_str:<20} | {name:<6} | {st:.3f} / {se:.3f}     | {nt:.3f} / {ne:.3f}    | {train_time:.1f}s / {test_time:.2f}s")
            current_models[name] = model

        print("-" * 85)

        # Capture snapshots for visualization
        if i in target_indices:
            snapshots.append({
                'scale': ns,
                'snr': snr,
                'models': deepcopy(current_models),
                'data': {
                    'train': (x_train_norm, x_train_next_norm, s_train, s_train_next, d_train, d_train_next),
                    'test': (x_test_norm, x_test_next_norm, s_test, s_test_next, d_test, d_test_next)
                }
            })

    # Plot reconstructions
    plot_reconstructions_row(snapshots)

    return results, noise_scales, snapshots


# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == "__main__":
    # Define scales and target indices
    scales = np.linspace(0, 8.0, 9)
    target_indices = [0, len(scales) // 2, len(scales) - 1]

    # Run experiment
    results, scales, snapshots = run_experiment_fixed_signal(scales, target_indices, seed=seed)

    # Define models list for plotting
    models_list = ['VAE', 'AR', 'JEPA', 'VJEPA', 'BJEPA']

    # Plotting 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'VAE': '#d62728', 'AR': '#ff7f0e', 'JEPA': '#17becf', 'VJEPA': '#1f77b4', 'BJEPA': '#9467bd'}
    markers = {'VAE': 'o', 'AR': 'x', 'JEPA': 'v', 'VJEPA': 's', 'BJEPA': 'D'}

    # 1. Training Signal Recovery
    ax = axes[0, 0]
    for k in models_list:
        ax.plot(scales, results['signal_train'][k], label=k, color=colors[k],
                linestyle='--', marker=markers[k], linewidth=2, alpha=0.7)
    ax.set_ylabel(r'Train $R^2$', fontsize=16)
    ax.set_title(r'Signal Recovery (Training)', fontsize=18)
    ax.set_xlabel('Distractor Noise Scale', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 2. Training Distractor Recovery
    ax = axes[0, 1]
    for k in models_list:
        ax.plot(scales[1:], results['distractor_train'][k][1:], label=k, color=colors[k],
                linestyle='--', marker=markers[k], linewidth=2, alpha=0.7)
    ax.set_ylabel(r'Train $R^2$', fontsize=16)
    ax.set_title(r'Distractor Recovery (Training)', fontsize=18)
    ax.set_xlabel('Distractor Noise Scale', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 3. Test Signal Recovery
    ax = axes[1, 0]
    for k in models_list:
        ax.plot(scales, results['signal_test'][k], label=k, color=colors[k],
                linestyle='-', marker=markers[k], linewidth=2)
    ax.set_ylabel(r'Test $R^2$', fontsize=16)
    ax.set_title(r'Signal Recovery (Test) - Generalization', fontsize=18)
    ax.set_xlabel('Distractor Noise Scale', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 4. Test Distractor Recovery
    ax = axes[1, 1]
    for k in models_list:
        ax.plot(scales[1:], results['distractor_test'][k][1:], label=k, color=colors[k],
                linestyle='-', marker=markers[k], linewidth=2)
    ax.set_ylabel(r'Test $R^2$', fontsize=16)
    ax.set_title(r'Distractor Recovery (Test)', fontsize=18)
    ax.set_xlabel('Distractor Noise Scale', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 85)
    print("Experiment complete!")
    print("=" * 85)

    # Print summary
    print("\n" + "=" * 85)
    print("KEY IMPLEMENTATION NOTES:")
    print("=" * 85)
    print("""
1. VAE: Uses encoder output z_t = enc_mu(x_t), evaluated against s_t (CURRENT signal)
   - VAE is a static model that represents current state

2. AR: Uses encoder output z_t = enc(x_t), evaluated against s_{t+1} (NEXT signal)
   - AR predicts x_{t+1} from z_t, so z_t implicitly represents prediction of future

3. JEPA: Uses PREDICTOR output z_pred = predictor(encoder(x_t)), evaluated against s_{t+1}
   - Predictor explicitly predicts Z_{t+1}

4. VJEPA: Uses predictor MEAN p_mu = pred_mu(encoder(x_t)), evaluated against s_{t+1}
   - Probabilistic predictor outputs distribution over Z_{t+1}

5. BJEPA: Uses HARD FUSION (Product of Experts) at inference:
   - Combines dynamics expert (pred_mu, pred_logvar) with prior (prior_mu, prior_logvar)
   - Returns fused posterior mean, evaluated against s_{t+1}
   - Training still uses soft fusion via KL regularization
""")

"""# End."""
