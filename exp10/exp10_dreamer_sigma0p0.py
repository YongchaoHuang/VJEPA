"""
Experiment 10: VJEPA-MPC on DM Control Suite — Cheetah-run
===========================================================
Condition : MODEL=DREAMER   SIGMA=0.0  SEED=0
Generated : standalone single-condition script (do not edit manually;
            re-run _gen_exp10_scripts.py if the template changes)
===========================================================
Instructions
  1. Open a new Colab notebook with an A100 GPU runtime.
  2. Install dependencies in the first cell:
         !pip install -q dm-control
  3. Paste this entire file into the second cell and run.
  4. Drive is mounted automatically; results persist at:
         /content/drive/MyDrive/VJEPA_experiments/exp10_dmc_vjepa_mpc/
  5. After completion share the checkpoint file:
         ckpt_cheetah_dreamer_s0.0_seed0.json

Comparison grid (18 conditions = 9 cheetah + 9 walker):
  Models : VJEPA-MPC (VICReg+NLL),  JEPA-MPC (VICReg only),  Dreamer-lite (pixel ELBO)
  Sigma  : 0.0 (clean),  0.5 (mild),  1.0 (noisy)
  Task   : Cheetah-run  (this script)
  Steps  : 500,000 env steps  (~7 h on A100)

IMPORTANT — do not change any hyperparameter in this file.
All 9 scripts must be byte-identical except for MODEL_TYPE / SIGMA.
"""
import os, sys, time, json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── OpenGL rendering backend — MUST be set before dm_control is imported ──────
# EGL uses the NVIDIA GPU driver directly: no apt packages, no ldconfig needed.
# OSMesa (CPU fallback) requires both PYOPENGL_PLATFORM and ldconfig after
# apt-get — two steps the original osmesa setup missed, causing the NoneType
# glGetError crash. EGL is the correct choice for any Colab GPU runtime.
_IN_COLAB = os.path.isdir('/content')
if _IN_COLAB:
    # Set env vars before any OpenGL/dm_control import.
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ.pop('PYOPENGL_PLATFORM', None)  # must be unset for EGL
    # PyOpenGL selects its rendering platform on first import and caches a
    # singleton for the process lifetime.  If a previous cell already imported
    # OpenGL with PYOPENGL_PLATFORM=osmesa, that singleton persists even after
    # clearing the env var.  Evicting all OpenGL/dm_control modules from
    # sys.modules forces a fresh import (and fresh platform selection) below.
    _stale = [m for m in sys.modules if m.startswith(('OpenGL', 'dm_control'))]
    for _m in _stale:
        del sys.modules[_m]
    os.system('apt-get install -y --quiet patchelf > /dev/null 2>&1')
else:
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

# ── Colab / A100 setup ────────────────────────────────────────────────────────
# Run in Colab as a notebook cell (paste full script, set args below main()).
# Each run (~2–4 h on A100) saves to OUT_DIR after every eval checkpoint,
# so multiple Colab sessions can accumulate results across models/sigma/seeds.
# Mount Drive first and set GDRIVE_DIR for persistent storage between sessions.
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_DIR   = None    # override: e.g. '/content/drive/MyDrive/VJEPA/exp10'
_DRIVE_ROOT  = '/content/drive/MyDrive'
_DRIVE_AVAIL = os.path.isdir(_DRIVE_ROOT)

# ── Auto-mount Google Drive (Colab only) ──────────────────────────────────────
# Results from a 7-hour run are lost if saved only to /content/.
# Mount Drive now so OUT_DIR resolves to a persistent path before training starts.
if _IN_COLAB and not _DRIVE_AVAIL:
    try:
        from google.colab import drive as _gdrive
        print("Mounting Google Drive for persistent result storage...")
        _gdrive.mount('/content/drive', force_remount=False)
        _DRIVE_AVAIL = os.path.isdir(_DRIVE_ROOT)
        if _DRIVE_AVAIL:
            print(f"  Drive mounted OK → results will persist at {_DRIVE_ROOT}/VJEPA_experiments/exp10_dmc_vjepa_mpc/")
        else:
            print("  WARNING: mount completed but path not found — check Drive permissions.")
    except Exception as _e:
        print(f"\n{'!'*70}")
        print(f"  WARNING: Google Drive auto-mount failed: {_e}")
        print(f"  Results will be saved to /content/ ONLY and will be LOST on disconnect.")
        print(f"  To fix: run the cell below before starting, then re-run this cell:")
        print(f"      from google.colab import drive; drive.mount('/content/drive')")
        print(f"{'!'*70}\n")
elif _IN_COLAB and _DRIVE_AVAIL:
    print(f"Drive already mounted → results will persist at {_DRIVE_ROOT}/VJEPA_experiments/exp10_dmc_vjepa_mpc/")

USE_AMP    = True    # bfloat16 autocast — ~30–50% faster CEM + training on A100
# Auto-disable AMP if the GPU doesn't support bfloat16 (requires A100/H100)
_BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_AMP   = USE_AMP and _BF16_OK

try:
    from dm_control import suite as dmc_suite
except ImportError:
    sys.exit("dm_control not found. Run: pip install dm-control")

# ── Output directory: Drive (auto) > /content > local ─────────────────────────
try:
    _local_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'new experiment results', 'exp10_dmc_vjepa_mpc')
except NameError:   # __file__ is not defined in Jupyter/Colab notebook cells
    _local_out = os.path.join(os.getcwd(), 'new experiment results', 'exp10_dmc_vjepa_mpc')
def _resolve_out():
    if GDRIVE_DIR:
        return GDRIVE_DIR
    if _DRIVE_AVAIL:
        return os.path.join(_DRIVE_ROOT, 'VJEPA_experiments', 'exp10_dmc_vjepa_mpc')
    return '/content/exp10_dmc_vjepa_mpc' if _IN_COLAB else _local_out

OUT_DIR = _resolve_out()
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fixed condition — DO NOT MODIFY ──────────────────────────────────────────
MODEL_TYPE = 'dreamer'   # 'vjepa', 'jepa', or 'dreamer'
SIGMA      = 0.0       # background Gaussian noise σ  (0.0 / 0.5 / 1.0)
SEED       = 0             # kept identical across all 9 conditions
TASK       = 'cheetah'     # 'cheetah' → cheetah/run  |  'walker' → walker/walk


# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_H, IMG_W    = 64, 64   # pixel observation size
IMG_SHAPE       = (3, IMG_H, IMG_W)   # channels-first for PyTorch
Z_DIM           = 256
PRED_HIDDEN     = 512

LR_WORLD        = 1e-4     # world model (encoder + predictor)
LR_REWARD       = 1e-3     # reward model
BATCH           = 512      # doubled for A100 (40 GB VRAM)
BUFFER_SIZE     = 100_000
EMA_TAU         = 0.995

BETA_VJEPA      = 0.01     # KL weight (not used in main loss — kept for compatibility)
BETA_DREAMER    = 0.1      # KL weight for Dreamer RSSM

# CEM planning
CEM_HORIZON     = 10
CEM_SAMPLES     = 1024     # doubled for A100 — better planning quality
CEM_ELITES      = 100      # 10% of CEM_SAMPLES
CEM_ITERS       = 3
CEM_MIN_STD     = 0.05
DISCOUNT        = 0.99

# Online training loop
RANDOM_STEPS    = 20_000   # random exploration before training — longer warm-up
                            # gives the replay buffer reward variance before the
                            # reward model starts training (5k was too few: almost
                            # all transitions had r≈0, collapsing the reward model)
TOTAL_STEPS     = 500_000
UPDATE_FREQ     = 2        # world model updates per env step
EVAL_FREQ       = 25_000   # evaluate every N steps
N_EVAL_EPISODES = 5        # episodes per evaluation

device = None   # set in main()


# ── DM Control environment wrapper ────────────────────────────────────────────
class DMCEnv:
    """
    Thin wrapper around dm_control.suite that:
      1. Renders 64×64×3 pixel observations
      2. Adds Gaussian noise to background pixels
      3. Returns CHW tensors (channels-first)
    """
    def __init__(self, domain, task, sigma_noise=0.0, seed=0, camera_id=0):
        self.env      = dmc_suite.load(domain, task,
                                        task_kwargs={'random': seed})
        self.sigma    = sigma_noise
        self.cam      = camera_id
        # Two separate RNGs so the action-selection sequence is identical across
        # σ values.  A shared RNG would advance at different rates for σ=0
        # (no randn calls) vs σ>0 (randn call per step), making random
        # exploration trajectories incomparable across noise conditions.
        self._noise_rng  = np.random.RandomState(seed + 7777)  # pixel noise
        self._action_rng = np.random.RandomState(seed + 8888)  # random action sampling
        spec          = self.env.action_spec()
        self.a_min    = spec.minimum.astype(np.float32)
        self.a_max    = spec.maximum.astype(np.float32)
        self.a_dim    = int(np.prod(spec.shape))

    def reset(self):
        ts = self.env.reset()
        return self._render()

    def step(self, action):
        action = np.clip(action, self.a_min, self.a_max)
        ts     = self.env.step(action)
        obs    = self._render()
        reward = float(ts.reward) if ts.reward is not None else 0.0
        done   = ts.last()
        return obs, reward, done

    def _render(self):
        frame = self.env.physics.render(
            height=IMG_H, width=IMG_W, camera_id=self.cam
        ).astype(np.float32) / 255.0   # [H, W, 3]
        if self.sigma > 0:
            noise = self._noise_rng.randn(*frame.shape).astype(np.float32) * self.sigma
            frame = np.clip(frame + noise, 0.0, 1.0)
        return frame.transpose(2, 0, 1)   # [3, H, W]  (CHW)

    def sample_random_action(self):
        return self._action_rng.uniform(self.a_min, self.a_max).astype(np.float32)


# ── Replay buffer ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxsize, obs_shape, a_dim):
        self.maxsize = maxsize
        self.obs  = np.zeros((maxsize, *obs_shape), dtype=np.float32)
        self.nobs = np.zeros((maxsize, *obs_shape), dtype=np.float32)
        self.acts = np.zeros((maxsize, a_dim),       dtype=np.float32)
        self.rews = np.zeros(maxsize,                dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs):
        self.obs [self.ptr] = obs
        self.nobs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.ptr  = (self.ptr + 1) % self.maxsize
        self.size = min(self.size + 1, self.maxsize)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, batch)
        return (torch.FloatTensor(self.obs [idx]),
                torch.FloatTensor(self.acts[idx]),
                torch.FloatTensor(self.rews[idx]),
                torch.FloatTensor(self.nobs[idx]))


# ── Shared CNN encoder/decoder ────────────────────────────────────────────────
# 64×64 → 32→16→8→4  (4 stride-2 conv layers)
CNN_FLAT = 256 * 4 * 4   # 4096

def make_enc():
    """3×64×64 → 256×4×4 → linear → z_dim."""
    return nn.Sequential(
        nn.Conv2d(3,   32, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32,  64, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128,256, 4, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(CNN_FLAT, Z_DIM)
    )

def make_dec():
    """z_dim → 256×4×4 → deconv → 3×64×64."""
    return nn.Sequential(
        nn.Linear(Z_DIM, CNN_FLAT), nn.ReLU(),
        nn.Unflatten(1, (256, 4, 4)),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d( 32,   3, 4, stride=2, padding=1), nn.Sigmoid()
    )


# ── World models ───────────────────────────────────────────────────────────────
class VJEPAAgent(nn.Module):
    """
    VJEPA-MPC world model.
    Encoder: CNN → z.
    Predictor: (z, a) → (mu, logvar) for z_{t+1}.
    Training: VICReg + stop-grad NLL uncertainty calibration (no pixel reconstruction).
    """
    def __init__(self, a_dim):
        super().__init__()
        self.enc      = make_enc()
        self.pred_mu  = nn.Sequential(
            nn.Linear(Z_DIM + a_dim, PRED_HIDDEN), nn.ReLU(),
            nn.Linear(PRED_HIDDEN, Z_DIM))
        self.pred_lv  = nn.Sequential(
            nn.Linear(Z_DIM + a_dim, PRED_HIDDEN), nn.ReLU(),
            nn.Linear(PRED_HIDDEN, Z_DIM))
        self.tgt      = make_enc()   # EMA target encoder
        for p in self.tgt.parameters(): p.requires_grad_(False)

    def forward(self, obs_t, a_t, obs_n):
        """Returns (pred_mu, pred_lv, tgt_z) for loss computation."""
        z_t  = self.enc(obs_t)
        za   = torch.cat([z_t, a_t], dim=-1)
        p_mu = self.pred_mu(za)
        p_lv = self.pred_lv(za).clamp(-10, 10)
        with torch.no_grad(): zt_n = self.tgt(obs_n)
        return p_mu, p_lv, z_t, zt_n

    def ema_update(self):
        for p, tp in zip(self.enc.parameters(), self.tgt.parameters()):
            tp.data.mul_(EMA_TAU).add_(p.data, alpha=1 - EMA_TAU)

    @torch.no_grad()
    def encode(self, obs):
        self.eval()
        z = self.enc(obs)
        self.train()
        return z

    @torch.no_grad()
    def rollout_mean(self, z, actions):
        """
        Vectorized latent rollout (deterministic, using predictor mean).
        z:       [N, Z_DIM]
        actions: [N, H, a_dim]
        Returns: [N, H, Z_DIM]  (one z per step after action)
        """
        H = actions.shape[1]
        zs = []
        for h in range(H):
            za = torch.cat([z, actions[:, h]], dim=-1)
            z  = self.pred_mu(za)
            zs.append(z)
        return torch.stack(zs, dim=1)   # [N, H, Z_DIM]


class JEPAAgent(nn.Module):
    """
    JEPA-MPC ablation: deterministic predictor, VICReg only, no uncertainty head.
    """
    def __init__(self, a_dim):
        super().__init__()
        self.enc  = make_enc()
        self.pred = nn.Sequential(
            nn.Linear(Z_DIM + a_dim, PRED_HIDDEN), nn.ReLU(),
            nn.Linear(PRED_HIDDEN, Z_DIM))
        self.tgt  = make_enc()
        for p in self.tgt.parameters(): p.requires_grad_(False)

    def forward(self, obs_t, a_t, obs_n):
        z_t  = self.enc(obs_t)
        za   = torch.cat([z_t, a_t], dim=-1)
        p_z  = self.pred(za)
        with torch.no_grad(): zt_n = self.tgt(obs_n)
        return p_z, z_t, zt_n

    def ema_update(self):
        for p, tp in zip(self.enc.parameters(), self.tgt.parameters()):
            tp.data.mul_(EMA_TAU).add_(p.data, alpha=1 - EMA_TAU)

    @torch.no_grad()
    def encode(self, obs):
        self.eval()
        z = self.enc(obs)
        self.train()
        return z

    @torch.no_grad()
    def rollout_mean(self, z, actions):
        H = actions.shape[1]
        zs = []
        for h in range(H):
            za = torch.cat([z, actions[:, h]], dim=-1)
            z  = self.pred(za)
            zs.append(z)
        return torch.stack(zs, dim=1)


class DreamerAgent(nn.Module):
    """
    Dreamer-lite: pixel ELBO reconstruction world model.
    Encoder: shared CNN backbone → z_mu; small linear head → z_lv.
    (Same CNN parameter count as VJEPA/JEPA — decoder is the legitimate extra capacity.)
    Decoder reconstructs NOISY pixels from posterior z → forces encoder to model noise.
    Key: reconstruction loss on noisy pixels is what causes degradation at high σ.
    """
    def __init__(self, a_dim):
        super().__init__()
        self.enc         = make_enc()                   # shared CNN backbone → z mean
        self.enc_lv_head = nn.Linear(Z_DIM, Z_DIM)     # log-var head for posterior
        # prior_mu is the learned transition model (z_t, a_t) → z_{t+1} mean.
        # It is trained via the KL term and also used for CEM rollouts — one
        # unified network avoids the bug of planning with an untrained predictor.
        self.prior_mu    = nn.Sequential(
            nn.Linear(Z_DIM + a_dim, PRED_HIDDEN), nn.ReLU(),
            nn.Linear(PRED_HIDDEN, Z_DIM))
        self.prior_lv    = nn.Linear(Z_DIM + a_dim, Z_DIM)   # transition log-var
        self.dec         = make_dec()                          # pixel decoder

    def encode(self, obs):
        """Encode obs → sampled z via reparameterization (eval: use mean)."""
        mu = self.enc(obs)
        if self.training:
            lv = self.enc_lv_head(mu).clamp(-10, 10)
            return mu + (0.5 * lv).exp() * torch.randn_like(mu), mu, lv
        return mu, mu, torch.zeros_like(mu)

    def forward(self, obs_t, a_t, obs_n):
        """Returns tensors needed for ELBO loss."""
        z_t, _mu_t, _lv_t = self.encode(obs_t)
        za    = torch.cat([z_t, a_t], dim=-1)
        z_n, mu_n, lv_n = self.encode(obs_n)
        x_rec = self.dec(z_n)           # reconstruct NOISY obs from posterior z
        pr_mu = self.prior_mu(za)       # prior mean — trained via KL, used for CEM
        pr_lv = self.prior_lv(za).clamp(-10, 10)
        return x_rec, (mu_n, lv_n), (pr_mu, pr_lv)

    @torch.no_grad()
    def rollout_mean(self, z, actions):
        H = actions.shape[1]
        zs = []
        for h in range(H):
            za = torch.cat([z, actions[:, h]], dim=-1)
            z  = self.prior_mu(za)   # use the trained transition prior for planning
            zs.append(z)
        return torch.stack(zs, dim=1)


# ── Reward model ───────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    """Linear reward predictor trained on (z, r) pairs from replay buffer."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)   # [B]


# ── Loss functions ─────────────────────────────────────────────────────────────
def vicreg(x, y, lam_sim=25., lam_std=25., lam_cov=1.):
    loss = lam_sim * (x - y).pow(2).mean()
    for z in (x, y):
        std = z.std(dim=0)
        loss += lam_std / 2 * torch.relu(1.0 - std).mean()
        zc   = z - z.mean(0)
        N    = max(z.shape[0] - 1, 1)
        cov  = (zc.T @ zc) / N
        off  = cov.pow(2)
        off.fill_diagonal_(0.0)
        loss += lam_cov * off.sum() / z.shape[1]
    return loss


def vjepa_world_model_loss(agent, obs_t, a_t, obs_n):
    """VICReg + stop-gradient NLL for VJEPA; VICReg only for JEPA."""
    if isinstance(agent, VJEPAAgent):
        p_mu, p_lv, z_t, zt_n = agent(obs_t, a_t, obs_n)
        loss = vicreg(p_mu, zt_n)
        # Uncertainty calibration (stop-grad from enc and pred_mu)
        z_sg   = z_t.detach()
        za_sg  = torch.cat([z_sg, a_t], dim=-1)
        p_lv_c = agent.pred_lv(za_sg).clamp(-10, 10)
        pv     = p_lv_c.exp()
        res    = (zt_n.detach() - agent.pred_mu(za_sg).detach()).pow(2)
        loss   = loss + 0.1 * 0.5 * (pv.log() + res / pv).sum(-1).mean()
    elif isinstance(agent, JEPAAgent):
        p_z, z_t, zt_n = agent(obs_t, a_t, obs_n)
        loss = vicreg(p_z, zt_n)
    else:
        raise ValueError
    return loss


def dreamer_world_model_loss(agent, obs_t, a_t, obs_n, beta=BETA_DREAMER):
    """Pixel ELBO: reconstruction + KL (posterior || prior)."""
    x_rec, (mu_n, lv_n), (pr_mu, pr_lv) = agent(obs_t, a_t, obs_n)
    # Reconstruction on noisy obs_n (this is what forces encoder to model noise!)
    rec_loss = F.mse_loss(x_rec, obs_n)
    # KL(posterior || prior)
    vr  = (lv_n - pr_lv).exp()
    kl  = 0.5 * (vr + (pr_mu - mu_n).pow(2) / pr_lv.exp() - 1 - (lv_n - pr_lv)).sum(-1).mean()
    return rec_loss + beta * kl


# ── CEM planning ──────────────────────────────────────────────────────────────
@torch.no_grad()
def cem_plan(z_now, agent, reward_model, a_dim,
             horizon=CEM_HORIZON, n_samples=CEM_SAMPLES,
             n_elites=CEM_ELITES, n_iters=CEM_ITERS):
    """
    Vectorized CEM in latent space (Algorithm 1 from paper).
    z_now:  [Z_DIM]  current latent state
    Returns: best_action [a_dim]
    """
    # Initialize action distribution
    mu  = torch.zeros(horizon, a_dim, device=device)
    std = torch.ones (horizon, a_dim, device=device)

    _amp = torch.autocast('cuda', dtype=torch.bfloat16,
                          enabled=USE_AMP and device.type == 'cuda')
    for _ in range(n_iters):
        # Sample N action sequences: [N, H, a_dim]
        noise   = torch.randn(n_samples, horizon, a_dim, device=device)
        actions = (mu.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)

        with _amp:
            # Latent rollout: [N, H, Z_DIM]
            z_batch = z_now.unsqueeze(0).expand(n_samples, -1)  # [N, Z_DIM]
            z_seq   = agent.rollout_mean(z_batch, actions)        # [N, H, Z_DIM]

            # Compute discounted returns from reward model
            z_flat  = z_seq.reshape(n_samples * horizon, Z_DIM)  # [N*H, Z_DIM]
            r_flat  = reward_model(z_flat).reshape(n_samples, horizon)
        discounts = torch.tensor(
            [DISCOUNT ** h for h in range(horizon)], device=device)
        returns   = (r_flat.float() * discounts.unsqueeze(0)).sum(dim=1)  # fp32 for topk

        # Elite update — normalise returns so elite selection is meaningful even
        # when all returns are near zero (prevents effectively-random selection
        # during the early phase when the reward model hasn't learned yet)
        ret_std   = returns.std().clamp_min(1e-6)
        ret_norm  = (returns - returns.mean()) / ret_std
        elite_idx = ret_norm.topk(n_elites).indices           # [n_elites]
        elite_a   = actions[elite_idx]                        # [n_elites, H, a_dim]
        mu  = elite_a.mean(0)
        std = elite_a.std(0).clamp_min(CEM_MIN_STD)

    return mu[0]   # [a_dim] — execute first action only


# ── Online MBRL training step ──────────────────────────────────────────────────
def world_model_step(agent, opt_wm, obs_t, a_t, obs_n):
    """Single gradient step on world model."""
    _amp = torch.autocast('cuda', dtype=torch.bfloat16,
                           enabled=USE_AMP and device.type == 'cuda')
    with _amp:
        if isinstance(agent, DreamerAgent):
            loss = dreamer_world_model_loss(agent, obs_t, a_t, obs_n)
        else:
            loss = vjepa_world_model_loss(agent, obs_t, a_t, obs_n)
    opt_wm.zero_grad()
    loss.backward()
    # Dreamer KL can spike when prior_lv → -inf at high σ; clip to prevent NaN
    nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
    opt_wm.step()
    return loss.item()


def reward_model_step(agent, reward_model, opt_r, obs_t, a_t, rewards):
    """
    Single gradient step on linear reward predictor.
    Trains on the *predicted* next-state latent pred(enc(obs_t), a_t) so the
    reward model sees the same latent distribution that CEM uses at plan time
    (CEM applies reward_model to rollout_mean outputs, not to raw enc outputs).
    """
    _amp = torch.autocast('cuda', dtype=torch.bfloat16,
                           enabled=USE_AMP and device.type == 'cuda')
    with torch.no_grad(), _amp:
        z_t = agent.enc(obs_t)
        za  = torch.cat([z_t, a_t], dim=-1)
        if isinstance(agent, VJEPAAgent):
            z_pred = agent.pred_mu(za)
        elif isinstance(agent, JEPAAgent):
            z_pred = agent.pred(za)
        else:   # DreamerAgent — use trained transition prior (same as rollout_mean)
            z_pred = agent.prior_mu(za)
    # Normalise rewards per-batch: forces the model to learn *relative* reward
    # differences even when all absolute values are near zero (which happens
    # during early training when the agent mostly falls over).  CEM only needs
    # a ranking signal, so this does not hurt planning quality.
    r_std  = rewards.std().clamp_min(1e-6)
    r_norm = (rewards - rewards.mean()) / r_std
    r_pred = reward_model(z_pred.float())   # fp32 for stability
    loss   = F.mse_loss(r_pred, r_norm)
    opt_r.zero_grad(); loss.backward(); opt_r.step()
    return loss.item()


# ── Evaluate agent ─────────────────────────────────────────────────────────────
def evaluate(agent, reward_model, domain, task_name, sigma, seed,
             n_episodes=N_EVAL_EPISODES):
    """Run n_episodes with CEM planning; return mean total reward."""
    agent.eval()
    reward_model.eval()
    total_rewards = []
    for ep in range(n_episodes):
        env = DMCEnv(domain, task_name, sigma_noise=sigma, seed=seed + ep + 9999)
        a_dim = env.a_dim
        obs = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            z = agent.enc(obs_t).squeeze(0)
            action = cem_plan(z, agent, reward_model, a_dim)
            obs, rew, done = env.step(action.cpu().numpy())
            ep_return += rew
        total_rewards.append(ep_return)
    agent.train()
    reward_model.train()
    return np.mean(total_rewards)


# ── Drive checkpoint helpers ──────────────────────────────────────────────────
def _ckpt_tag(domain, model_type, sigma, seed):
    return f"{domain}_{model_type}_s{sigma}_seed{seed}"

def _flush_eval_ckpt(domain, model_type, sigma, seed, eval_steps, eval_returns):
    """
    Write per-run eval history JSON after every evaluation checkpoint.
    File: OUT_DIR/ckpt_<tag>.json — always up-to-date on Drive.
    Gives full learning-curve data even if Colab disconnects mid-run.
    """
    tag  = _ckpt_tag(domain, model_type, sigma, seed)
    path = os.path.join(OUT_DIR, f'ckpt_{tag}.json')
    with open(path, 'w') as f:
        json.dump({'domain': domain, 'model_type': model_type,
                   'sigma': sigma, 'seed': seed,
                   'eval_steps': eval_steps, 'eval_returns': eval_returns}, f)
    print(f"    [Ckpt] {path}  ({len(eval_steps)} evals so far)")


def _save_model_ckpt(domain, model_type, sigma, seed, step,
                     agent, reward_model, opt_wm, opt_r):
    """
    Save agent + reward-model state dicts so a disconnected run can be resumed.
    File: OUT_DIR/model_<tag>_step<N>.pt  (overwrites previous to save space).
    """
    tag  = _ckpt_tag(domain, model_type, sigma, seed)
    path = os.path.join(OUT_DIR, f'model_{tag}.pt')   # single file, overwritten each time
    torch.save({'step': step,
                'agent':        agent.state_dict(),
                'reward_model': reward_model.state_dict(),
                'opt_wm':       opt_wm.state_dict(),
                'opt_r':        opt_r.state_dict()}, path)
    print(f"    [Ckpt] model saved → {path}  (step {step})")


# ── Main training loop ─────────────────────────────────────────────────────────
def run_experiment(domain, task_name, model_type, sigma, seed, total_steps,
                   dry_run=False):
    """
    Full online MBRL loop.
    Returns: eval_steps [list], eval_returns [list]
    """
    np.random.seed(seed); torch.manual_seed(seed)

    env   = DMCEnv(domain, task_name, sigma_noise=sigma, seed=seed)
    a_dim = env.a_dim
    print(f"\n{'='*65}")
    print(f"  Task={domain}/{task_name}  Model={model_type.upper()}  "
          f"σ={sigma}  seed={seed}")
    print(f"  a_dim={a_dim}  obs_shape={IMG_SHAPE}  z_dim={Z_DIM}  "
          f"steps={total_steps}")
    print(f"{'='*65}")

    # Create agent
    if model_type == 'vjepa':
        agent = VJEPAAgent(a_dim).to(device)
    elif model_type == 'jepa':
        agent = JEPAAgent(a_dim).to(device)
    elif model_type == 'dreamer':
        agent = DreamerAgent(a_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    reward_model = RewardModel().to(device)
    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Agent parameters: {n_params:,}")

    opt_wm = optim.Adam(agent.parameters(), lr=LR_WORLD)
    opt_r  = optim.Adam(reward_model.parameters(), lr=LR_REWARD)

    buffer = ReplayBuffer(BUFFER_SIZE, IMG_SHAPE, a_dim)

    # ── Phase 1: Random exploration ────────────────────────────────────────────
    print(f"\n  Phase 1: random exploration ({RANDOM_STEPS} steps)...")
    obs = env.reset()
    for step in range(RANDOM_STEPS):
        action = env.sample_random_action()
        next_obs, rew, done = env.step(action)
        buffer.add(obs, action, rew, next_obs)
        obs = next_obs
        if done: obs = env.reset()
    print(f"  Buffer filled: {buffer.size} transitions")

    # ── Phase 2: Online MBRL ───────────────────────────────────────────────────
    print(f"  Phase 2: online MBRL...")
    eval_steps   = []
    eval_returns = []
    obs = env.reset()
    wm_losses = []; r_losses = []; step_times = []
    actual_steps = 100 if dry_run else (total_steps - RANDOM_STEPS)

    for step in range(actual_steps):
        t0 = time.time()

        # Encode + plan
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            z = agent.enc(obs_t).squeeze(0)
        action = cem_plan(z, agent, reward_model, a_dim)

        # Execute in environment
        next_obs, rew, done = env.step(action.cpu().numpy())
        buffer.add(obs, action.cpu().numpy(), rew, next_obs)
        obs = next_obs
        if done: obs = env.reset()

        # World model + reward model update
        if step % UPDATE_FREQ == 0 and buffer.size >= BATCH:
            o_t, a_t, r_t, o_n = buffer.sample(BATCH)
            o_t = o_t.to(device); a_t = a_t.to(device)
            r_t = r_t.to(device); o_n = o_n.to(device)
            wl = world_model_step(agent, opt_wm, o_t, a_t, o_n)
            rl = reward_model_step(agent, reward_model, opt_r, o_t, a_t, r_t)
            wm_losses.append(wl); r_losses.append(rl)
            # EMA update
            if hasattr(agent, 'ema_update'):
                agent.ema_update()

        step_times.append(time.time() - t0)

        global_step = RANDOM_STEPS + step + 1

        # Logging
        if (step + 1) % 5000 == 0:
            avg_wm = np.mean(wm_losses[-500:]) if wm_losses else float('nan')
            avg_r  = np.mean(r_losses [-500:]) if r_losses  else float('nan')
            avg_t  = np.mean(step_times[-500:]) * 1000
            eta_h  = (actual_steps - step - 1) * np.mean(step_times) / 3600
            print(f"    step {global_step:6d}/{total_steps}  "
                  f"wm_loss={avg_wm:.4f}  r_loss={avg_r:.4f}  "
                  f"step={avg_t:.1f}ms  ETA={eta_h:.1f}h")

        # Evaluation + immediate Drive flush
        if global_step % EVAL_FREQ == 0 or (dry_run and step == actual_steps - 1):
            ret = evaluate(agent, reward_model, domain, task_name,
                           sigma=sigma, seed=seed)
            eval_steps.append(global_step)
            eval_returns.append(ret)
            print(f"    *** EVAL step={global_step}  return={ret:.1f} ***")

            # Flush eval history and model state to Drive after every checkpoint.
            # If Colab disconnects, the last eval + model weights are preserved.
            _flush_eval_ckpt(domain, model_type, sigma, seed,
                             eval_steps, eval_returns)
            _save_model_ckpt(domain, model_type, sigma, seed, global_step,
                             agent, reward_model, opt_wm, opt_r)

    return eval_steps, eval_returns, agent, reward_model


# ── Plot helpers ───────────────────────────────────────────────────────────────
def load_results(task):
    path = os.path.join(OUT_DIR, f'results_{task}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_results(task, results):
    path = os.path.join(OUT_DIR, f'results_{task}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def plot_task(task, results, save_path):
    """Learning curve: episode return vs. env steps, one line per model×sigma."""
    sigmas = ['0.0', '0.5', '1.0']
    models = ['vjepa', 'jepa', 'dreamer']
    labels = {'vjepa': 'VJEPA-MPC', 'jepa': 'JEPA-MPC', 'dreamer': 'Dreamer-lite'}
    colours = {'vjepa': '#ff7f0e', 'jepa': '#1f77b4', 'dreamer': '#2ca02c'}
    lstyles = {'0.0': '-', '0.5': '--', '1.0': ':'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax_i, sigma in enumerate(sigmas):
        ax = axes[ax_i]
        for model in models:
            key = f'{model}_sigma{sigma}'
            if key not in results: continue
            seeds_data = results[key]   # list of (steps, returns) per seed
            # Align to common steps grid
            if not seeds_data: continue
            all_steps   = seeds_data[0][0]
            all_returns = np.array([d[1] for d in seeds_data])
            mean = all_returns.mean(0)
            sem  = all_returns.std(0) / max(len(all_returns) ** 0.5, 1)
            ax.plot(all_steps, mean, label=labels[model],
                    color=colours[model], linewidth=2.5)
            ax.fill_between(all_steps, mean - sem, mean + sem,
                            color=colours[model], alpha=0.15)
        ax.set_title(f'σ = {sigma}', fontsize=12)
        ax.set_xlabel('Environment steps', fontsize=11)
        if ax_i == 0: ax.set_ylabel('Episode return', fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'Exp. 10: {task} — VJEPA-MPC vs Baselines under Visual Noise\n'
                 '(Gaussian background distractor σ, higher = more noise)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_noise_robustness(results_ch, results_wk, save_path):
    """Bar chart: final return at σ=0,0.5,1.0 for each model × task."""
    sigmas = [0.0, 0.5, 1.0]
    models = ['vjepa', 'jepa', 'dreamer']
    labels = {'vjepa': 'VJEPA-MPC', 'jepa': 'JEPA-MPC', 'dreamer': 'Dreamer-lite'}
    colours = {'vjepa': '#ff7f0e', 'jepa': '#1f77b4', 'dreamer': '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task, results) in zip(axes, [('Cheetah-run', results_ch),
                                           ('Walker-walk',  results_wk)]):
        x = np.arange(len(sigmas))
        w = 0.25
        for mi, model in enumerate(models):
            vals, errs = [], []
            for sigma in sigmas:
                key  = f'{model}_sigma{sigma}'
                data = results.get(key, [])
                if data:
                    finals = [d[1][-1] if d[1] else 0.0 for d in data]
                    vals.append(np.mean(finals))
                    errs.append(np.std(finals))
                else:
                    vals.append(0.0); errs.append(0.0)
            ax.bar(x + mi * w, vals, w, label=labels[model],
                   color=colours[model], yerr=errs, capsize=4,
                   error_kw={'elinewidth': 1.2})
        ax.set_xticks(x + w)
        ax.set_xticklabels([f'σ={s}' for s in sigmas], fontsize=11)
        ax.set_ylabel('Final episode return', fontsize=11)
        ax.set_title(f'{task}', fontsize=12)
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Exp. 10: Noise Robustness — Final Return under Visual Distractors\n'
                 'VJEPA-MPC (predictive) vs. Dreamer-lite (generative) vs. JEPA-MPC (ablation)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  Z_DIM={Z_DIM}  |  AMP={USE_AMP}  |  "
          f"CEM(H={CEM_HORIZON},N={CEM_SAMPLES},K={CEM_ITERS})")
    print(f"Condition: model={MODEL_TYPE}  sigma={SIGMA}  seed={SEED}  task={TASK}")

    domain, task_name = {'cheetah': ('cheetah', 'run'),
                         'walker':  ('walker',  'walk')}[TASK]

    results = load_results(TASK)
    key     = f'{MODEL_TYPE}_sigma{SIGMA}'

    steps, returns, _, _ = run_experiment(
        domain, task_name, MODEL_TYPE, SIGMA, SEED, TOTAL_STEPS)

    if key not in results:
        results[key] = []
    results[key].append((steps, returns))
    save_results(TASK, results)

    plot_task(TASK, results,
              os.path.join(OUT_DIR, f'fig_dmc_{TASK}_return.pdf'))

    print(f"\n  ✓ {key}  seed={SEED}  "
          f"final_return={returns[-1] if returns else 'N/A':.1f}")


if __name__ == '__main__':
    main()
