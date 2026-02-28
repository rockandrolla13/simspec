"""
SimConfig: All simulation parameters with defaults from spec Section 19.

All parameters are documented with their spec equation references and realistic ranges.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimConfig:
    """
    Complete configuration for the RFQ simulator.

    All defaults match sim_spec_v2.pdf Section 19 (Complete Parameter Reference).
    """

    # =========================================================================
    # Simulation Control
    # =========================================================================
    T_days: int = 60
    """Horizon in trading days."""

    dt_minutes: float = 5.0
    """Price step size in minutes."""

    trading_hours: float = 8.0
    """Trading hours per day."""

    n_mc_paths: int = 500
    """Monte Carlo paths for batch runs."""

    seed: int = 42
    """Master random seed."""

    # =========================================================================
    # Price Process (Eq 1-3)
    # =========================================================================
    p0: float = 100.0
    """Initial price in dollars."""

    sigma_daily_bps: float = 50.0
    """Daily volatility in bps. IG: [20,80], HY: [50,200]."""

    kappa_daily: float = 0.02
    """Mean-reversion speed per day. IG: [0.005,0.05], HY: [0.01,0.10]."""

    p_bar: float = 100.0
    """Long-run mean price."""

    v_open: float = 0.5
    """Open volatility multiplier for intraday seasonality."""

    v_close: float = 0.3
    """Close volatility multiplier for intraday seasonality."""

    tau_v_hours: float = 0.75
    """Volatility seasonality width in hours."""

    # =========================================================================
    # Alpha Signal (Eq 4-7)
    # =========================================================================
    IC: float = 0.10
    """Information coefficient (signal-to-noise correlation). Typical: [0.03,0.15]."""

    IC_stress_mult: float = 0.4
    """IC multiplier in stressed regime (IC degrades)."""

    alpha_horizon_days: float = 10.0
    """Signal horizon H in days. Typical: [5,20]."""

    signal_refresh_hours: float = 24.0
    """Hours between signal updates."""

    p_calm_to_stress: float = 0.05
    """Daily transition probability calm→stressed."""

    p_stress_to_calm: float = 0.15
    """Daily transition probability stressed→calm. Stationary ~25% stress."""

    # =========================================================================
    # Position Target (Eq 8-9)
    # =========================================================================
    gamma: float = 1.0
    """Risk aversion coefficient."""

    q_max: int = 20
    """Maximum position in lots."""

    lot_size_mm: float = 1.0
    """Face value per lot in $MM."""

    # =========================================================================
    # RFQ Arrivals (Eq 10, Section 6)
    # =========================================================================
    rfq_rate_per_day: float = 15.0
    """Mean daily RFQ count. IG: [15,40], HY: [2,8]."""

    A_open: float = 0.8
    """Open activity multiplier for intraday seasonality."""

    A_close: float = 0.5
    """Close activity multiplier for intraday seasonality."""

    tau_f_hours: float = 0.75
    """RFQ seasonality width in hours."""

    flow_bias: float = 0.0
    """Directional bias δ_flow ∈ [-0.3, 0.3]. 0 = balanced."""

    size_mu: float = 0.5
    """Log-mean of size distribution."""

    size_sigma: float = 0.8
    """Log-std of size distribution."""

    size_max: int = 10
    """Maximum trade size in lots."""

    n_dealers_mean: float = 4.0
    """Mean competing dealers N̄. IG: [4,5], HY: [2,3]."""

    n_dealers_max: int = 8
    """Maximum competing dealers."""

    tox_a: float = 2.0
    """Toxicity Beta shape a. Beta(2,8) → mean 0.2, 90th pctile ~0.45."""

    tox_b: float = 8.0
    """Toxicity Beta shape b."""

    # =========================================================================
    # Observable Mid and Skew (Eq 11-12)
    # =========================================================================
    obs_lag_minutes: float = 15.0
    """Observation delay (staleness) in minutes."""

    obs_noise_bps: float = 5.0
    """Observation noise in bps."""

    skew_accuracy: float = 0.3
    """Skew model accuracy ρ_s ∈ [0,1]."""

    # =========================================================================
    # Lean Computation (Eq 13-15)
    # =========================================================================
    lambda_base_bps: float = 2.0
    """Base lean per unit inventory gap, in bps."""

    sigma_ref_bps: float = 50.0
    """Reference daily vol for normalization."""

    kappa_urgency: float = 1.0
    """Urgency multiplier at signal expiry."""

    kappa_convexity: float = 0.1
    """Convexity per lot of inventory gap."""

    kappa_limit: float = 10.0
    """Soft limit penalty strength."""

    theta_limit: float = 0.7
    """Fraction of q_max where soft penalty activates."""

    # =========================================================================
    # Competitor Model (Eq 16-20) - CRITICAL
    # =========================================================================
    dealer_bias_std_bps: float = 3.0
    """σ_b: Cross-dealer lean dispersion in bps."""

    markup_base_bps: float = 8.0
    """m̄: Baseline dealer markup in bps. IG: [3,8], HY: [15,40]."""

    markup_N_bps: float = -1.0
    """m_N: Markup shift per additional dealer (negative = tighter)."""

    markup_size_bps: float = 2.0
    """m_size: Markup shift per log-lot (positive = wider for large)."""

    markup_tox_bps: float = 10.0
    """m_tox: Markup shift per unit toxicity (positive = wider for toxic)."""

    markup_noise_bps: float = 3.0
    """σ_m: Per-quote markup noise in bps."""

    quote_noise_bps: float = 2.0
    """σ_q: Dealer model error / quote noise in bps."""

    respond_base: float = 0.85
    """r̄: Base response rate."""

    respond_size: float = 0.05
    """r_size: Size impact on response (reduces for larger)."""

    respond_tox: float = 0.1
    """r_tox: Toxicity impact on response (reduces for toxic)."""

    # =========================================================================
    # Win-Rate Estimation (Eq 21)
    # =========================================================================
    winrate_est_error: float = 0.0
    """Degradation factor on trader's ĉ, β̂. 0 = perfect calibration."""

    m_max_bps: float = 30.0
    """Maximum markup grid extent in bps."""

    m_grid_bps: float = 0.1
    """Markup grid resolution in bps."""

    # =========================================================================
    # Post-Fill / Adverse Selection (Eq 25)
    # =========================================================================
    adverse_move_bps: float = 5.0
    """σ_adverse: Adverse selection scale in bps."""

    # =========================================================================
    # Hybrid Exit (Eq 26-27)
    # =========================================================================
    aggress_window_hours: float = 8.0
    """Δt_aggress: Time before signal expiry to start aggressing. Default = 1 trading day."""

    aggress_halfspread_bps: float = 8.0
    """c_aggress: Half-spread cost for aggressive execution."""

    aggress_impact_bps: float = 2.0
    """c_impact: Market impact per √lot."""

    # =========================================================================
    # Street Lean (Eq 28-32)
    # =========================================================================
    street_lean_mean_rev: float = 0.1
    """θ_b: Mean-reversion speed of street lean."""

    street_lean_vol_bps: float = 2.0
    """σ_b: Volatility of street lean process."""

    street_lean_eq: float = 0.0
    """b̄_eq: Equilibrium street lean."""

    street_obs_noise: float = 0.5
    """Noise on each street lean proxy."""

    street_proxy_weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    """[w1, w2, w3]: Weights for bid-ask asymmetry, flow imbalance, ETF premium."""

    # =========================================================================
    # Hedging (Eq 33-34)
    # =========================================================================
    hedge_cds_on: bool = False
    """Enable CDS hedging."""

    hedge_etf_on: bool = False
    """Enable ETF hedging."""

    hedge_tsy_on: bool = False
    """Enable Treasury hedging."""

    rho_cds: float = 0.85
    """CDS-bond correlation."""

    rho_etf: float = 0.70
    """ETF-bond correlation."""

    rho_tsy: float = 0.50
    """Treasury-bond correlation."""

    cds_cost_bps: float = 2.0
    """CDS execution cost in bps."""

    etf_cost_bps: float = 1.0
    """ETF execution cost in bps."""

    tsy_cost_bps: float = 0.5
    """Treasury execution cost in bps."""

    # =========================================================================
    # Carry (Eq 38)
    # =========================================================================
    coupon_bps: float = 500.0
    """Annual coupon in bps of face (5% = 500 bps)."""

    # =========================================================================
    # Derived Properties
    # =========================================================================
    @property
    def minutes_per_day(self) -> float:
        """Total trading minutes per day."""
        return self.trading_hours * 60.0

    @property
    def n_steps_per_day(self) -> int:
        """Number of price steps per trading day."""
        return int(self.minutes_per_day / self.dt_minutes)

    @property
    def n_steps(self) -> int:
        """Total number of price steps in simulation."""
        return self.T_days * self.n_steps_per_day

    @property
    def total_minutes(self) -> float:
        """Total simulation time in minutes."""
        return self.T_days * self.minutes_per_day

    @property
    def alpha_horizon_minutes(self) -> float:
        """Alpha horizon in minutes."""
        return self.alpha_horizon_days * self.minutes_per_day

    @property
    def sigma_per_step(self) -> float:
        """
        Volatility per price step, in dollar terms.

        Eq 3: σ_per_step = σ_daily_bps/10000 * p0 * sqrt(dt_min/480)
        """
        return (self.sigma_daily_bps / 10000.0) * self.p0 * (self.dt_minutes / 480.0) ** 0.5

    @property
    def kappa_per_step(self) -> float:
        """Mean-reversion speed per price step."""
        return self.kappa_daily * self.dt_minutes / 480.0

    @property
    def rfq_rate_per_minute(self) -> float:
        """RFQ arrival rate per minute (for Poisson process)."""
        return self.rfq_rate_per_day / self.minutes_per_day

    def validate(self) -> None:
        """Check parameter validity."""
        assert self.T_days > 0, "T_days must be positive"
        assert self.dt_minutes > 0, "dt_minutes must be positive"
        assert 0 <= self.IC <= 1, "IC must be in [0, 1]"
        assert 0 < self.IC_stress_mult <= 1, "IC_stress_mult must be in (0, 1]"
        assert self.q_max > 0, "q_max must be positive"
        assert -0.5 <= self.flow_bias <= 0.5, "flow_bias must be in [-0.5, 0.5]"
        assert 0 <= self.skew_accuracy <= 1, "skew_accuracy must be in [0, 1]"
        assert self.theta_limit > 0 and self.theta_limit < 1, "theta_limit must be in (0, 1)"
        assert len(self.street_proxy_weights) == 3, "street_proxy_weights must have 3 elements"
