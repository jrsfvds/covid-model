# =============================================================================
# COVID-MODEL — CORE METHODOLOGY MODULE
# Louis Lecas
#
# "covid_model_local_Experimental_Beta_Relative_differentialEvolution"
#
# This file holds the modelling methodology ONLY: data loading, beta
# functions, compartmental models (SIR/SEIR/SEIRS), the ODE solver wrapper,
# and the differential_evolution-based global parameter fit.
#
# It is imported as a module by app.py (Dash UI). There is no __main__
# entry point here on purpose — all interaction happens through the UI.
#
# N optimization and global optimization using scipy.optimize.differential_evolution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
# =============================================================================

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# =============================================================================
# DEFAULT PARAMETER SELECTION
# These are the defaults used unless overridden by the caller (e.g. the Dash
# UI sliders). Kept here so the module remains runnable/testable standalone.
# =============================================================================

# Path to the CSV file from the RKI - this one is compressed to only "Anzahl_Fall" category
# and is quicker, if all the categories are kept it takes much longer
FILE_PATH = r"data/Aktuell_Deutschland_SarsCov2_Infektionen_total.csv"

# ---- The fixed (non-optimised) epidemiological parameters ----------------------------------------
#
# gamma: recovery rate  = 1 / infectious_period  (10-day infectious period)
#
#   Plausible average according to:
#       Byrne, A.W., McEvoy, D., Collins, A.B., Hunt, K., Casey, M., Barber, A., Butler, F., Griffin, J., Lane, E.A., McAloon, C., O'Brien, K., Wall, P., Walsh, K.A. and More, S.J."Inferred duration of infectious period of SARS-CoV-2: rapid scoping review and analysis of available evidence for asymptomatic and symptomatic COVID-19 cases.", 2019, BMJ open, doi:https://doi.org/10.1136/bmjopen-2020-039856.
#           Accessed 21 May 2026
#       Drain, P.K., Dalmat, R.R., Hao, L., Bemer, M.J., Budiawan, E., Morton, J.F., Ireton, R.C., Hsiang, T.-Y., Marfatia, Z., Prabhu, R., Woosley, C., Gichamo, A., Rechkina, E., Hamilton, D., Montaño, M., Cantera, J.L., Ball, A.S., Golez, I., Smith, E. and Greninger, A.L. "Duration of viral infectiousness and correlation with symptoms and diagnostic testing in non-hospitalized adults during acute SARS-CoV-2 infection: A longitudinal cohort study.", 2023,  Journal of Clinical Virology, doi:https://doi.org/10.1016/j.jcv.2023.105420.
#           Accessed 21 May 2026
#
# sigma: incubation rate = 1 / incubation_period (5-day incubation, only SEIR/SEIRS)
#
#   Plausible estimate according to:
#       CDC. "Clinical Presentation." COVID-19, 29 Oct. 2024, www.cdc.gov/covid/hcp/clinical-care/covid19-presentation.html#cdc_generic_section_4-incubation-period.
#           Accessed 21 May 2026.
#
# omega: waning rate    = 1 / immunity_duration  (180-day immunity, SEIRS only)
#
#   Plausible estimate according to:
#       Marcotte, Harold, et al. "Immunity to SARS-CoV-2 up to 15 Months after Infection." IScience, vol. 25, no. 2, Feb. 2022, p. 103743, https://doi.org/10.1016/j.isci.2022.103743.
#           Accessed 11 Nov. 2022.
#       CDC. "About Reinfection." COVID-19, 15 July 2024, www.cdc.gov/covid/about/reinfection.html.
#           Accessed 21 May 2026.
#

GAMMA = 1 / 10
SIGMA = 1 / 5
OMEGA = 1 / 180

# ---- Population bounds -------------------------------------------------------
# N represents the *effective interacting population*, not the national total!
# No start guess as optimizer is Global
N_MIN = 1_000           # under 1000 would be ridiculous
N_MAX = 83_000_000      # approx. German population

# ---- Differential evolution settings -----------------------------------------
# search parameters for optimization (used as UI defaults — can be overridden
# by the caller for interactive use, since maxiter=5000 is too slow for a
# live Dash callback)
DE_MAXITER = 5000
DE_POPSIZE = 15
DE_TOL = 1e-12
DE_SEED = 42
DE_WORKERS = -1
DE_POLISH = True       # local L-BFGS-B optimization after global search


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(file_path: str = FILE_PATH) -> pd.DataFrame:
    """Load the RKI case data CSV, parsing the report date column."""
    return pd.read_csv(file_path, parse_dates=["Meldedatum"])


# =============================================================================
# 7-day SMOOTHING
# =============================================================================

def smooth_series(data: np.ndarray, window: int = 7) -> np.ndarray:
    return (
        pd.Series(data)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .values
    )

# =============================================================================
# BETA FUNCTIONS
# Each function returns the transmission rate beta at time t.
# =============================================================================

def beta_constant(t, b0):
    return np.clip(b0, 0, 2)

def beta_linear(t, b0, b1):
    return np.clip(b0 + b1 * t, 0, 2)

def beta_polynomial(t, b0, b1, b2):
    return np.clip(b0 + b1 * t + b2 * t ** 2, 0, 2)

def beta_time(t, params, mode: str) -> float:
    dispatch = {
        "constant":   beta_constant,
        "linear":     beta_linear,
        "polynomial": beta_polynomial,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown beta mode: {mode!r}")
    return dispatch[mode](t, *params)

# =============================================================================
# BETA BOUNDS - very broad
# Used directly by differential_evolution — no initial guess required.
# =============================================================================

def get_beta_bounds(beta_mode: str, n: int) -> list:

    bounds_map = {
        "constant":   [(0, 100)],
        "linear":     [(0, 100), (-10, 10)],
        "polynomial": [(0, 100), (-10, 10), (-1, 1)],
    }
    return bounds_map[beta_mode]

# =============================================================================
# COMPARTMENTAL MODELS
# =============================================================================

def SIR(t, y, beta_params, beta_mode, gamma, N):
    S, I, R = y
    beta  = beta_time(t, beta_params, beta_mode)
    dSdt  = -beta * S * I / N
    dIdt  =  beta * S * I / N - gamma * I
    dRdt  =  gamma * I
    return [dSdt, dIdt, dRdt]

def SEIR(t, y, beta_params, beta_mode, sigma, gamma, N):
    S, E, I, R = y
    beta  = beta_time(t, beta_params, beta_mode)
    dSdt  = -beta * S * I / N
    dEdt  =  beta * S * I / N - sigma * E
    dIdt  =  sigma * E - gamma * I
    dRdt  =  gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def SEIRS(t, y, beta_params, beta_mode, sigma, gamma, omega, N):
    S, E, I, R = y
    beta  = beta_time(t, beta_params, beta_mode)
    dSdt  = -beta * S * I / N + omega * R
    dEdt  =  beta * S * I / N - sigma * E
    dIdt  =  sigma * E - gamma * I
    dRdt  =  gamma * I - omega * R
    return [dSdt, dEdt, dIdt, dRdt]

MODEL_MAP = {"SIR": SIR, "SEIR": SEIR, "SEIRS": SEIRS}

# =============================================================================
# HELPERS
# =============================================================================

def build_args(model_func, beta_params, beta_mode, sigma, gamma, omega, N):
    """Putting model arguments in the order expected by each model function because they are all different."""
    if model_func is SIR:
        return (beta_params, beta_mode, gamma, N)
    elif model_func is SEIR:
        return (beta_params, beta_mode, sigma, gamma, N)
    else:  # this one is then SEIRS
        return (beta_params, beta_mode, sigma, gamma, omega, N)

def build_initial_conditions(model_func, N, I0, E0, R0_init=0):
    """
    Create starting conditions (y0) such that S0 = N - E0 - I0 - R0.
    Returns None if it doesn't work (S0 <= 0).
    """
    if model_func is SIR:
        S0 = N - I0 - R0_init
        return [S0, I0, R0_init] if S0 > 0 else None
    else:
        S0 = N - (E0 or 0) - I0 - R0_init
        return [S0, E0, I0, R0_init] if S0 > 0 else None

def extract_I(model_func, solution: np.ndarray) -> np.ndarray:
    """Return the infectious compartment I from the solvers output (from the array)."""
    return solution[1] if model_func is SIR else solution[2]

def run_model(model_func, y0, t_eval, args, tight=False):
    """Solve the ODE system using RK45."""
    rtol = 1e-6 if tight else 1e-3  # Tolerance - Error control
    atol = 1e-8 if tight else 1e-5
    sol = solve_ivp(
        fun=lambda t, y: model_func(t, y, *args),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    return sol.y

# =============================================================================
# PARAMETER FITTING — GLOBAL OPTIMISATION
#
# Optimizer: Differential Evolution
# ----------------------------
# L-BFGS-B is a local gradient-based method and is highly sensitive to its
# starting point. When the loss "landscape" (n-dimensional representation of loss) is multimodal (with many peaks as is typical with
# combined beta / N / I0 optimisation), it can converge to very bad
# local minima depending on the initial guess — making the result
# unreliable and reducing it's utility for this specific scenario.
#
# differential_evolution uses a population of candidate solutions and
# explores the parameter space without requiring an initial guess.
# This removes the need for heuristics such as the ridiculous "N_start = 5 × max(I)", replaced with N as an additional optimized Parameter.
#
# On N
# ----------
# N is the *effective interacting population* — the sub-population within which
# the epidemic actually spreads. It is NOT the national total.
# =============================================================================

def fit_parameters(model_func, t_train, I_train, beta_mode,
                    sigma=SIGMA, gamma=GAMMA, omega=OMEGA,
                    n_min=N_MIN, n_max=N_MAX,
                    de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE,
                    de_tol=DE_TOL, de_seed=DE_SEED,
                    de_workers=DE_WORKERS, de_polish=DE_POLISH,
                    progress_callback=None):
    """
    Fit beta/N/I0(/E0) parameters via global optimisation (differential_evolution).

    progress_callback: optional callable(intermediate_result) -> None,
    forwarded to differential_evolution's `callback` kwarg so a caller
    (e.g. a Dash app) can report progress during a long-running fit.
    """

    n         = len(t_train)
    R0_init   = 0  # No initial Recovered
    beta_bnds = get_beta_bounds(beta_mode, n)

    # Build the full bounds vector depending on model type:
    #   [*beta_bounds, N_bound, I0_bound]            — SIR
    #   [*beta_bounds, N_bound, I0_bound, e0_factor] — SEIR/SEIRS
    #
    # e0_factor: E0 = e0_factor * I0 (ratio avoids hard-coding E0, but is kind of still super heuristic)
    #
    # I0 upper bound is set to half of n_max rather than n_max itself, because otherwise the model doesn't work anyways.
    # (Also so that the polishing step with L-BFGS-B cannot push I0 above N in the final result.)
    I0_MAX = n_max * 0.5
    if model_func is SIR:
        bounds = beta_bnds + [(n_min, n_max), (1, I0_MAX)]
    else:
        bounds = beta_bnds + [(n_min, n_max), (1, I0_MAX), (0.0, 5.0)]

    # bounds here are : Beta stuff, N, I0 and E0-factor (No R because set to 0)

    # ------------------------------------------------------------------
    # Loss function: Mean Squared Error between model I(t) and RKI
    # ------------------------------------------------------------------
    def loss(params):
        if model_func is SIR:
            *beta_params, N_opt, I0 = params
            E0 = None
        else:
            *beta_params, N_opt, I0, e0_factor = params
            E0 = e0_factor * I0

        if I0 >= N_opt:
            return 1e12  # infeasible

        y0_local = build_initial_conditions(model_func, N_opt, I0, E0, R0_init)
        if y0_local is None:
            return 1e12

        args = build_args(
            model_func, tuple(beta_params), beta_mode,
            sigma, gamma, omega, N_opt
        )
        sol     = run_model(model_func, y0_local, t_train, args, tight=False)
        I_model = extract_I(model_func, sol)
        return np.mean((I_model - I_train) ** 2)

    # ------------------------------------------------------------------
    # Global search with optional local polishing
    # ------------------------------------------------------------------
    de_kwargs = dict(
        bounds        = bounds,
        maxiter       = de_maxiter,
        popsize       = de_popsize,
        tol           = de_tol,
        seed          = de_seed,
        workers       = de_workers,
        polish        = de_polish,    # local L-BFGS-B applied on best result
        mutation      = (0.5, 1.5),   # The mutation constant. In the literature this is also known as differential weight.
                                       # If specified as a float it should be in the range [0, 2). If specified as a tuple (min, max) dithering is employed.
                                       # Dithering randomly changes the mutation constant on a generation by generation basis. The mutation constant for that generation is taken from U[min, max).
                                       # Dithering can help speed convergence significantly. Increasing the mutation constant increases the search radius, but will slow down convergence.
        recombination = 0.9,
    )
    if progress_callback is not None:
        de_kwargs["callback"] = progress_callback

    result = differential_evolution(loss, **de_kwargs)

    # ------------------------------------------------------------------
    # Unpacking optimised parameters
    # The local polishing step (L-BFGS-B) could push values
    # outside the bounds, so a clamp stops that.
    # ------------------------------------------------------------------
    if model_func is SIR:
        *beta_params, N_opt, I0_opt = result.x
        E0_opt = None
    else:
        *beta_params, N_opt, I0_opt, e0_factor_opt = result.x
        e0_factor_opt = float(np.clip(e0_factor_opt, 0.0, 5.0))
        E0_opt        = e0_factor_opt * I0_opt

    # Ensure I0 (+ E0 for SEIR/SEIRS) leaves room for S0 > 0
    overhead = I0_opt + (E0_opt if E0_opt is not None else 0.0)
    if overhead >= N_opt:
        scale  = (N_opt - 1.0) / overhead
        I0_opt = I0_opt * scale
        if E0_opt is not None:
            E0_opt = E0_opt * scale

    return np.array(beta_params), N_opt, I0_opt, E0_opt