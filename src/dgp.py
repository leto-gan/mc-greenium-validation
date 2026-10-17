"""Data-generating process (DGP) for the Monte Carlo design-validation study.

Calibrated to the sample dimensions of Section 4.1 of the research proposal
("Greenium Fragility and Climate Shocks", C. Gan):

- 150 matched green-conventional bond pairs. Each pair belongs to one issuer
  that has issued one green and one conventional bond, so issuer-level
  clustering has 150 clusters of two bonds.
- T = 800 trading days (approx. 3.2 years), abstracting a 2021-2024 window
  that spans the SFDR Level 1 -> Level 2 transition.
- K = 40 regional climate shocks, randomly timed, each hitting one of 12
  geographic regions for an elevated-response window of 10 trading days.
- Regime variable in two implementations:
    R1 (binary):    EU-domiciled issuer x post-Level-2 (t > 400)
    R2 (intensity): 2022 year-end Article 8/9 fund ownership share,
                    pre-determined at t <= 400, ~ Beta(2, 5).

All quantities are in basis points (bp) of yield spread.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel dimensions (Table M1 of the appendix)
# ---------------------------------------------------------------------------
N_ISSUERS = 150
N_BONDS = 2 * N_ISSUERS
T_DAYS = 800
K_SHOCKS = 40
N_REGIONS = 12
POST_START = 400            # SFDR Level 2 effective for t > POST_START
SHOCK_WINDOW = 10           # trading days of elevated response per shock

# ---------------------------------------------------------------------------
# Calibration (Table M1)
# ---------------------------------------------------------------------------
TAU = 8.5                   # true differential shock response of the greenium
                            # (bp), anchored to Del Giudice et al. (2025)
KAPPA = 4.0                 # common shock response of conventional spreads (bp)
MU_GREEN = -5.0             # baseline greenium level (bp), Zerbib (2019)
SD_ISSUER_BASE = 15.0       # cross-issuer spread dispersion (bp)
SIGMA_TOTAL = 15.0          # total daily residual sd per bond (bp)
REGION_CORR = 0.2           # share of residual variance common within region
VAR_ISSUER_RE = 25.0        # time-invariant issuer random-effect variance
RHO_AR = 0.3                # AR(1) coefficient of the idiosyncratic residual
DRIFT_BP_PER_YEAR = 3.0     # S4/W4 confounder: EU greenium drift after Level 2
DAYS_PER_YEAR = 250.0
DELIST_SHARE = 0.15         # S5: share of bonds delisting mid-sample

DESIGN_SEED = 20261006      # fixed cross-sectional design across all scenarios

# ---------------------------------------------------------------------------
# Scenario matrix (Section 3.4 of the framework; W* = intensity-regime DGPs)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "S1": dict(regime="binary",    phi=-4.0),                    # benchmark
    "S2": dict(regime="binary",    phi=0.0),                     # size
    "S3": dict(regime="binary",    phi=-2.0),                    # power, weak
    "S6": dict(regime="binary",    phi=-6.0),                    # power, strong
    "S4": dict(regime="binary",    phi=-4.0, confound=True),     # EU drift
    "S5": dict(regime="binary",    phi=-4.0, unbalanced=True),   # delistings
    "W1": dict(regime="intensity", phi=-4.0),
    "W2": dict(regime="intensity", phi=0.0),
    "W3": dict(regime="intensity", phi=-2.0),
    "W6": dict(regime="intensity", phi=-6.0),
    "W4": dict(regime="intensity", phi=-4.0, confound=True),
    "S3b": dict(regime="binary",    phi=-3.0),                   # decision-rule
    "W3b": dict(regime="intensity", phi=-3.0),                   # threshold
}

EVENT_STUDY_SCENARIOS = ["S1", "S4", "W4"]
ES_BIN_EDGES = list(range(-30, 35, 5))   # 5-day bins from -30 to +30


class Design:
    """Fixed cross-sectional structure, common to all scenarios and reps."""

    def __init__(self, seed=DESIGN_SEED):
        self.n_bonds = N_BONDS
        rng = np.random.default_rng(seed)
        iss = np.arange(N_ISSUERS)
        self.region = rng.integers(0, N_REGIONS, N_ISSUERS)
        self.eu = rng.random(N_ISSUERS) < 0.5
        self.base = rng.normal(0.0, SD_ISSUER_BASE, N_ISSUERS)

        self.bond_issuer = np.repeat(iss, 2)
        self.green = np.zeros(N_BONDS, dtype=bool)
        self.green[0::2] = True                       # one green bond per issuer
        self.bond_region = self.region[self.bond_issuer]
        self.bond_eu = self.eu[self.bond_issuer]
        self.mu = self.base[self.bond_issuer].copy()
        self.mu[self.green] += MU_GREEN

        # Pre-determined Article 8/9 ownership intensity (green bonds only)
        self.w = np.zeros(N_BONDS)
        ng = int(self.green.sum())
        self.w[self.green] = rng.beta(2.0, 5.0, ng)

        # Bond covariates with selection: green bonds are higher rated,
        # shorter maturity, larger size (mimics the observed selection that
        # motivates the matching estimators E4/E5).
        self.rating = np.empty(N_BONDS, dtype=int)    # 0 best .. 2 worst
        self.rating[self.green] = rng.choice(3, ng, p=[0.60, 0.30, 0.10])
        self.rating[~self.green] = rng.choice(3, N_BONDS - ng,
                                              p=[0.30, 0.40, 0.30])
        self.maturity = np.empty(N_BONDS)
        self.maturity[self.green] = rng.uniform(3.0, 10.0, ng)
        self.maturity[~self.green] = rng.uniform(2.0, 15.0, N_BONDS - ng)
        self.logsize = np.empty(N_BONDS)
        self.logsize[self.green] = rng.normal(20.3, 0.7, ng)
        self.logsize[~self.green] = rng.normal(19.8, 0.9, N_BONDS - ng)

        # Matching weights are design-fixed: computed once, reused in all reps
        from estimators import entropy_balance_weights, cem_weights
        self.w_eb, self.eb_converged = entropy_balance_weights(self)
        self.w_cem, self.keep_cem = cem_weights(self)

        # median ownership intensity among green bonds (event-study split)
        self.w_median = float(np.median(self.w[self.green]))

    @property
    def pair_spread_index(self):
        """Issuer index for each bond pair (0..149)."""
        return np.arange(N_ISSUERS)


def make_design(seed=DESIGN_SEED):
    return Design(seed)


def simulate_rep(cfg, design, seed):
    """Draw one Monte Carlo replication for scenario config `cfg`.

    Returns dict with y (B x T spread matrix), shock (B x T bool), events
    (list of (day, region)), mask (B x T bool; False = missing/delisted).
    """
    rng = np.random.default_rng(seed)
    B, T = N_BONDS, T_DAYS

    # --- shock schedule -------------------------------------------------
    d_events = rng.integers(35, T - SHOCK_WINDOW - 35, K_SHOCKS)
    r_events = rng.integers(0, N_REGIONS, K_SHOCKS)
    shock = np.zeros((B, T), dtype=bool)
    for d, r in zip(d_events, r_events):
        shock[design.bond_region == r, d:d + SHOCK_WINDOW] = True

    # --- market factor with volatility clustering (absorbed by day FE) --
    z = rng.standard_normal(T)
    lam = np.empty(T)
    h = 1.0
    prev = 0.0
    for t in range(T):
        # GARCH(1,1): sd of lam approx 2 bp, alpha+beta = 0.95 (stationary)
        h = 0.20 + 0.10 * prev ** 2 + 0.85 * h
        lam[t] = np.sqrt(h) * z[t]
        prev = lam[t]

    # --- residuals: issuer RE + region-day common + AR(1) idiosyncratic -
    var_tot = SIGMA_TOTAL ** 2
    sd_region = np.sqrt(REGION_CORR * var_tot)
    sd_idio_stat = np.sqrt((1.0 - REGION_CORR) * var_tot - VAR_ISSUER_RE)
    sd_innov = sd_idio_stat * np.sqrt(1.0 - RHO_AR ** 2)
    u = rng.normal(0.0, np.sqrt(VAR_ISSUER_RE), N_ISSUERS)
    reg_noise = rng.normal(0.0, sd_region, (N_REGIONS, T))
    innov = rng.normal(0.0, sd_innov, (B, T))
    eps = np.empty((B, T))
    eps[:, 0] = innov[:, 0] / np.sqrt(1.0 - RHO_AR ** 2)
    for t in range(1, T):
        eps[:, t] = RHO_AR * eps[:, t - 1] + innov[:, t]
    eps += reg_noise[design.bond_region]
    eps += u[design.bond_issuer][:, None]

    # --- conditional mean -----------------------------------------------
    post = np.arange(T) > POST_START
    g = design.green
    y = design.mu[:, None] + lam[None, :] + eps
    if cfg["regime"] == "binary":
        R = design.bond_eu[:, None] & post[None, :]
        diff_response = TAU + cfg["phi"] * R.astype(float)
    else:
        diff_response = TAU + cfg["phi"] * design.w[:, None]
    y += shock * (KAPPA + g[:, None] * diff_response)

    # --- S4/W4 confounder: EU-specific greenium drift after Level 2 -----
    if cfg.get("confound"):
        t = np.arange(T)
        drift = np.where(post,
                         DRIFT_BP_PER_YEAR * (t - POST_START) / DAYS_PER_YEAR,
                         0.0)
        y[g & design.bond_eu] += drift[None, :]

    # --- S5 unbalanced panel: random mid-sample delistings ---------------
    mask = np.ones((B, T), dtype=bool)
    if cfg.get("unbalanced"):
        n_delist = int(round(DELIST_SHARE * B))
        idx = rng.choice(B, n_delist, replace=False)
        for b in idx:
            d0 = int(rng.integers(POST_START, T - 20))
            mask[b, d0:] = False

    return dict(y=y, shock=shock,
                events=list(zip(d_events.tolist(), r_events.tolist())),
                mask=mask)
