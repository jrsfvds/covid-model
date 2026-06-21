import time
import traceback

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import webbrowser

# =====================================================
# STANDALONE NOTE
# =====================================================
# This file is self-contained: every model/fitting function that used to
# live in covid_model_local_Experimental_Beta_Relative_differentialEvolution.py
# has been inlined below, so app.py no longer imports that module and can be
# dropped in anywhere app_SafetyCopy.py used to live (same relative data
# path, same `app`/`server` contract, same `python app.py` entry point).

# =====================================================
# BOOTSTRAP GUARD
# =====================================================
# Everything that builds the real app (data loading, Dash construction,
# layout, callbacks) happens inside _build_app(). If ANY of that throws —
# for any reason, on any host — we still end up with a working `server`
# attribute, because WSGI hosts (gunicorn etc.) look that up by name right
# after importing this file. Without this guard, an exception anywhere in
# that chain prevents `server` from ever being defined, and the host reports
# a confusing "module 'app' has no attribute 'server'" instead of the real
# cause. With this guard, the real traceback is printed to the logs AND
# rendered directly on the page, so it's diagnosable without redeploying.

BOOTSTRAP_ERROR = None

# =====================================================
# DATA PATH (relative — matches app_SafetyCopy.py)
# =====================================================
FILE_PATH = r"data/Aktuell_Deutschland_SarsCov2_Infektionen_total.csv"

# =====================================================
# FIXED EPIDEMIOLOGICAL PARAMETERS
# =====================================================
# gamma: recovery rate  = 1 / infectious_period  (10-day infectious period)
# sigma: incubation rate = 1 / incubation_period (5-day incubation, SEIR/SEIRS only)
# omega: waning rate    = 1 / immunity_duration  (180-day immunity, SEIRS only)
GAMMA = 1 / 10
SIGMA = 1 / 5
OMEGA = 1 / 180

# ---- Population bounds --------------------------------------------------
# N represents the *effective interacting population*, not the national
# total. No start guess is needed since the optimizer is global.
N_MIN = 1_000          # under 1000 would be ridiculous
N_MAX = 83_000_000     # approx. German population

# ---- Differential evolution defaults -------------------------------------
# (maxiter / popsize are also exposed as sliders in the UI; these are the
# fallback defaults plus the settings the UI does not expose.)
DE_MAXITER = 80
DE_POPSIZE = 10
DE_TOL = 1e-12
DE_SEED = 42
DE_WORKERS = 1          # forced to 1 in the Dash callback: the loss closure
                         # below is not picklable for multiprocessing workers
DE_POLISH = True        # local L-BFGS-B polish after the global search


# =====================================================
# DATA LOADING
# =====================================================
def load_data(file_path):
    """Load the RKI case data CSV. Raises if the file is missing/malformed,
    so callers can surface the error in the UI instead of crashing on import."""
    return pd.read_csv(file_path, parse_dates=["Meldedatum"])


# =====================================================
# 7-DAY SMOOTHING
# =====================================================
def smooth_series(data: np.ndarray, window: int = 7) -> np.ndarray:
    return (
        pd.Series(data)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .values
    )


# =====================================================
# BETA FUNCTIONS
# Each function returns the transmission rate beta at time t.
# =====================================================
def beta_constant(t, b0):
    return np.clip(b0, 0, 2)


def beta_linear(t, b0, b1):
    return np.clip(b0 + b1 * t, 0, 2)


def beta_polynomial(t, b0, b1, b2):
    return np.clip(b0 + b1 * t + b2 * t ** 2, 0, 2)


def beta_time(t, params, mode: str):
    dispatch = {
        "constant": beta_constant,
        "linear": beta_linear,
        "polynomial": beta_polynomial,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown beta mode: {mode!r}")
    return dispatch[mode](t, *params)


# =====================================================
# BETA BOUNDS — very broad, used directly by
# differential_evolution (no initial guess required).
# =====================================================
def get_beta_bounds(beta_mode: str, n: int) -> list:
    bounds_map = {
        "constant": [(0, 100)],
        "linear": [(0, 100), (-10, 10)],
        "polynomial": [(0, 100), (-10, 10), (-1, 1)],
    }
    return bounds_map[beta_mode]


# =====================================================
# COMPARTMENTAL MODELS
# =====================================================
def SIR(t, y, beta_params, beta_mode, gamma, N):
    S, I, R = y
    beta = beta_time(t, beta_params, beta_mode)
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def SEIR(t, y, beta_params, beta_mode, sigma, gamma, N):
    S, E, I, R = y
    beta = beta_time(t, beta_params, beta_mode)
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def SEIRS(t, y, beta_params, beta_mode, sigma, gamma, omega, N):
    S, E, I, R = y
    beta = beta_time(t, beta_params, beta_mode)
    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return [dSdt, dEdt, dIdt, dRdt]


MODEL_MAP = {"SIR": SIR, "SEIR": SEIR, "SEIRS": SEIRS}


# =====================================================
# HELPERS
# =====================================================
def build_args(model_func, beta_params, beta_mode, sigma, gamma, omega, N):
    """Model arguments in the order expected by each model function (they differ)."""
    if model_func is SIR:
        return (beta_params, beta_mode, gamma, N)
    elif model_func is SEIR:
        return (beta_params, beta_mode, sigma, gamma, N)
    else:  # SEIRS
        return (beta_params, beta_mode, sigma, gamma, omega, N)


def build_initial_conditions(model_func, N, I0, E0, R0_init=0):
    """Create y0 such that S0 = N - E0 - I0 - R0. Returns None if S0 <= 0."""
    if model_func is SIR:
        S0 = N - I0 - R0_init
        return [S0, I0, R0_init] if S0 > 0 else None
    else:
        S0 = N - (E0 or 0) - I0 - R0_init
        return [S0, E0, I0, R0_init] if S0 > 0 else None


def extract_I(model_func, solution: np.ndarray) -> np.ndarray:
    """Return the infectious compartment I from the solver output."""
    return solution[1] if model_func is SIR else solution[2]


def run_model(model_func, y0, t_eval, args, tight=False):
    """Solve the ODE system using RK45."""
    rtol = 1e-6 if tight else 1e-3
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


# =====================================================
# PARAMETER FITTING — GLOBAL OPTIMISATION
#
# differential_evolution uses a population of candidate solutions and
# explores the parameter space without requiring an initial guess, which
# is more robust against local minima than a gradient method (L-BFGS-B)
# started from a single heuristic guess. N, I0 (and E0 for SEIR/SEIRS)
# are optimised jointly with beta(t) rather than fixed by hand.
#
# N is the *effective interacting population* — the sub-population within
# which the epidemic actually spreads. It is NOT the national total.
# =====================================================
def fit_parameters(model_func, t_train, I_train, beta_mode,
                    sigma=SIGMA, gamma=GAMMA, omega=OMEGA,
                    n_min=N_MIN, n_max=N_MAX,
                    de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE,
                    de_tol=DE_TOL, de_seed=DE_SEED,
                    de_workers=DE_WORKERS, de_polish=DE_POLISH):

    n = len(t_train)
    R0_init = 0  # no initial recovered
    beta_bnds = get_beta_bounds(beta_mode, n)

    # Full bounds vector depending on model type:
    #   [*beta_bounds, N_bound, I0_bound]            — SIR
    #   [*beta_bounds, N_bound, I0_bound, e0_factor]  — SEIR/SEIRS
    #
    # e0_factor: E0 = e0_factor * I0 (ratio avoids hard-coding E0).
    # I0 upper bound is half of n_max (otherwise S0 can't stay positive,
    # and it keeps the L-BFGS-B polish step from pushing I0 above N).
    I0_MAX = n_max * 0.5
    if model_func is SIR:
        bounds = beta_bnds + [(n_min, n_max), (1, I0_MAX)]
    else:
        bounds = beta_bnds + [(n_min, n_max), (1, I0_MAX), (0.0, 5.0)]

    # ------------------------------------------------------------------
    # Loss function: Mean Squared Error between model I(t) and real data
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
        sol = run_model(model_func, y0_local, t_train, args, tight=False)
        I_model = extract_I(model_func, sol)
        return np.mean((I_model - I_train) ** 2)

    # ------------------------------------------------------------------
    # Global search with optional local polishing
    # ------------------------------------------------------------------
    result = differential_evolution(
        loss,
        bounds=bounds,
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=de_tol,
        seed=de_seed,
        workers=de_workers,
        polish=de_polish,        # local L-BFGS-B applied on best result
        mutation=(0.5, 1.5),     # dithered mutation constant, speeds convergence
        recombination=0.9,
    )

    # ------------------------------------------------------------------
    # Unpack optimised parameters. The local polishing step (L-BFGS-B)
    # could push values outside the bounds, so a clamp stops that.
    # ------------------------------------------------------------------
    if model_func is SIR:
        *beta_params, N_opt, I0_opt = result.x
        E0_opt = None
    else:
        *beta_params, N_opt, I0_opt, e0_factor_opt = result.x
        e0_factor_opt = float(np.clip(e0_factor_opt, 0.0, 5.0))
        E0_opt = e0_factor_opt * I0_opt

    # Ensure I0 (+ E0 for SEIR/SEIRS) leaves room for S0 > 0
    overhead = I0_opt + (E0_opt if E0_opt is not None else 0.0)
    if overhead >= N_opt:
        scale = (N_opt - 1.0) / overhead
        I0_opt = I0_opt * scale
        if E0_opt is not None:
            E0_opt = E0_opt * scale

    return np.array(beta_params), N_opt, I0_opt, E0_opt


# =====================================================
# DASH APP
# =====================================================
def _build_app():
    """Build and return (app, server). May raise — caller handles fallback."""

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    server = app.server

    # -------------------------------------------------
    # DATEN LADEN
    # -------------------------------------------------
    data_load_error = None
    try:
        df = load_data(FILE_PATH)
    except Exception as exc:  # noqa: BLE001 - intentionally broad: surface in UI, not just logs
        data_load_error = f"{exc}"
        print("ERROR loading RKI data:\n" + traceback.format_exc())
        df = pd.DataFrame({"Meldedatum": pd.to_datetime([]), "AnzahlFall": []})

    BETA_MODE_OPTIONS = [
        {"label": "Konstant", "value": "constant"},
        {"label": "Linear", "value": "linear"},
        {"label": "Polynomial", "value": "polynomial"},
    ]

    # Safe defaults for the date picker in case data loading failed (df is empty)
    default_start = df["Meldedatum"].min() if len(df) else "2020-01-01"
    default_end = df["Meldedatum"].max() if len(df) else "2020-12-31"

    app.layout = dbc.Container(fluid=True, children=[

        # --- Info Modal ---
        dbc.Modal(
            [
                dbc.ModalHeader("Informationen zur Simulation"),
                dbc.ModalBody(
                    dbc.Container(
                        "Dieses Modell nutzt ein dynamisches Kompartimentmodell (SIR, SEIR oder SEIRS), "
                        "dessen Parameter — inklusive der effektiven Population N, I₀ (und bei SEIR/SEIRS E₀) "
                        "sowie der Transmissionsrate β(t) — global über differential_evolution "
                        "(scipy.optimize) optimiert werden, statt lokal über L-BFGS-B mit einem Startwert. "
                        "Das macht die Anpassung robuster gegenüber lokalen Minima, dauert dafür aber länger. "
                        "Da N nicht mehr von Hand gesetzt wird, gibt es dafür keinen Slider mehr — der Wert wird "
                        "nach jedem Optimierungslauf unten rechts angezeigt. "
                        "Klicke auf 'Optimierung starten', um einen neuen Fit mit den aktuell gewählten "
                        "Einstellungen zu berechnen.",
                        style={"maxHeight": "60vh", "overflowY": "auto"}
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button("Schließen", id="close_info", n_clicks=0)
                ),
            ],
            id="modal_info",
            is_open=False
        ),

        # --- Überschrift ---
        dbc.Row(dbc.Col(html.H2("Dynamisches Modell COVID (Deutschland gesamt)", className="text-center my-3"))),
        dbc.Row(dbc.Col(html.P(
            "Globale Parameteroptimierung (differential_evolution) — N, I₀, E₀ und β(t) werden automatisch bestimmt.",
            className="text-center text-muted"
        ))),

        *([dbc.Row(dbc.Col(dbc.Alert(
            [
                html.Strong("Daten konnten nicht geladen werden: "),
                data_load_error,
            ],
            color="danger",
            className="mx-3"
        )))] if data_load_error else []),

        # --- Haupt-Layout ---
        dbc.Row([

            # Sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([

                        html.Label("Zeitraum"),
                        dcc.DatePickerRange(
                            id="date_range",
                            start_date=default_start,
                            end_date=default_end
                        ),

                        html.Br(), html.Br(),
                        html.Label("Modelltyp"),
                        dcc.RadioItems(
                            id="model_type",
                            options=["SIR", "SEIR", "SEIRS"],
                            value="SEIR",
                            inline=True
                        ),

                        html.Br(),
                        html.Label("β(t) Modus"),
                        dcc.RadioItems(
                            id="beta_mode",
                            options=BETA_MODE_OPTIONS,
                            value="polynomial",
                            inline=True
                        ),

                        html.Br(),
                        html.Label("Fit Datenbasis"),
                        dcc.RadioItems(
                            id="fit_mode",
                            options=[{"label": "Rohdaten", "value": "raw"},
                                     {"label": "7-Tage-Mittel", "value": "smooth"}],
                            value="smooth",
                            inline=True
                        ),

                        html.Br(),
                        html.Label("Train Split (%)"),
                        dcc.Slider(id="split_ratio", min=10, max=90, step=5, value=50,
                                   marks=None, tooltip={"placement": "bottom", "always_visible": False}),

                        html.Br(),
                        html.Label("R(t) anzeigen"),
                        dcc.Checklist(
                            id="show_rt",
                            options=[{"label": " Ja", "value": "yes"}],
                            value=[],
                            inline=True
                        ),

                        html.Hr(),
                        html.Label("Optimierungsqualität (Differential Evolution)"),

                        html.Div("Iterationen (maxiter)", className="mt-2 small text-muted"),
                        dcc.Slider(id="de_maxiter", min=10, max=500, step=10, value=80,
                                   marks={10: "10", 100: "100", 250: "250", 500: "500"},
                                   tooltip={"placement": "bottom", "always_visible": False}),

                        html.Div("Populationsgröße (popsize)", className="mt-2 small text-muted"),
                        dcc.Slider(id="de_popsize", min=4, max=30, step=1, value=10,
                                   marks={4: "4", 15: "15", 30: "30"},
                                   tooltip={"placement": "bottom", "always_visible": False}),

                        html.Div(
                            "Höhere Werte = genauerer Fit, aber deutlich längere Laufzeit.",
                            className="small text-muted mt-1"
                        ),

                        html.Br(),
                        dbc.Button("Optimierung starten", id="run_optimization",
                                   n_clicks=0, color="primary", className="w-100"),

                        html.Br(), html.Br(),
                        dbc.Button("Info zur Simulation", id="open_info",
                                   n_clicks=0, color="info", outline=True, className="w-100"),

                        html.Hr(),
                        html.Div(id="fit_results", className="small"),

                    ])
                ])
            ], width=3),

            # Plot
            dbc.Col(
                dcc.Loading(
                    id="loading_plot",
                    type="circle",
                    children=dcc.Graph(id="simulation_plot", style={"height": "75vh"})
                ),
                width=9
            )

        ])
    ])

    # =====================================================
    # Callback für Simulation Plot (läuft nur per Button-Klick)
    # =====================================================
    @app.callback(
        Output("simulation_plot", "figure"),
        Output("fit_results", "children"),
        Input("run_optimization", "n_clicks"),
        State("date_range", "start_date"),
        State("date_range", "end_date"),
        State("model_type", "value"),
        State("beta_mode", "value"),
        State("split_ratio", "value"),
        State("fit_mode", "value"),
        State("show_rt", "value"),
        State("de_maxiter", "value"),
        State("de_popsize", "value"),
        prevent_initial_call=False,
    )
    def update_simulation(n_clicks, start, end, model_type, beta_mode,
                           split_ratio, fit_mode, show_rt, de_maxiter, de_popsize):

        # Vor dem ersten Klick: leerer Plot mit Hinweistext
        if not n_clicks:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                annotations=[dict(
                    text="Einstellungen wählen und auf 'Optimierung starten' klicken.",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=16)
                )]
            )
            return fig, ""

        if data_load_error or len(df) == 0:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                annotations=[dict(
                    text="Keine Daten verfügbar — siehe Fehlermeldung oben.",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
                )]
            )
            return fig, ""

        # ------------------------------------------------------------------
        # Daten filtern
        # ------------------------------------------------------------------
        df_period = df[(df["Meldedatum"] >= start) & (df["Meldedatum"] <= end)]
        daily = df_period.groupby("Meldedatum")["AnzahlFall"].sum().sort_index()
        dates = daily.index
        I_real = daily.values.astype(float)

        if len(I_real) < 10:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                annotations=[dict(
                    text="Zu wenige Datenpunkte im gewählten Zeitraum (mind. 10 nötig).",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
                )]
            )
            return fig, ""

        # 7-Tage-Mittel
        I_smooth = smooth_series(I_real, 7)
        I_used = I_smooth if fit_mode == "smooth" else I_real.copy()

        # Split Train/Test
        split_idx = int(len(I_real) * split_ratio / 100)
        split_idx = max(2, min(split_idx, len(I_real) - 1))
        I_train = I_used[:split_idx]
        I_test = I_real[split_idx:]

        t_train = np.arange(len(I_train), dtype=float)
        t_total = np.arange(len(I_real), dtype=float)

        model_func = MODEL_MAP[model_type]

        # ------------------------------------------------------------------
        # Globale Optimierung (differential_evolution)
        # ------------------------------------------------------------------
        t0 = time.time()
        beta_params, N_opt, I0_opt, E0_opt = fit_parameters(
            model_func, t_train, I_train, beta_mode,
            sigma=SIGMA, gamma=GAMMA, omega=OMEGA,
            n_min=N_MIN, n_max=N_MAX,
            de_maxiter=de_maxiter, de_popsize=de_popsize,
            de_tol=DE_TOL, de_seed=DE_SEED,
            de_workers=1, de_polish=DE_POLISH,
        )
        fit_seconds = time.time() - t0

        # ------------------------------------------------------------------
        # Vollständige Simulation mit optimierten Parametern
        # ------------------------------------------------------------------
        y0 = build_initial_conditions(model_func, N_opt, I0_opt, E0_opt, R0_init=0)
        args = build_args(model_func, beta_params, beta_mode, SIGMA, GAMMA, OMEGA, N_opt)
        sol = run_model(model_func, y0, t_total, args, tight=True)
        I_model = extract_I(model_func, sol)

        # ------------------------------------------------------------------
        # Fehlerberechnung (Testfenster)
        # ------------------------------------------------------------------
        I_model_test = I_model[split_idx:]
        rmse = np.sqrt(mean_squared_error(I_test, I_model_test))
        mae = mean_absolute_error(I_test, I_model_test)

        # ------------------------------------------------------------------
        # R(t) — effektive Reproduktionszahl
        # ------------------------------------------------------------------
        S_t = sol[0]
        beta_t = beta_time(t_total, beta_params, beta_mode)
        R_t = beta_t / GAMMA * (S_t / N_opt)

        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=I_real, mode="markers", name="Rohdaten", opacity=0.4))
        fig.add_trace(go.Scatter(x=dates, y=I_smooth, mode="lines", name="7-Tage-Mittel"))
        fig.add_trace(go.Scatter(x=dates, y=I_model, mode="lines", name="Modell", line=dict(color="red", width=3)))

        if "yes" in show_rt:
            fig.add_trace(go.Scatter(x=dates, y=R_t, mode="lines", name="R(t)", yaxis="y2",
                                      line=dict(color="green", width=3)))

        cut_date = dates[split_idx]

        fig.add_vrect(x0=cut_date, x1=dates[-1], fillcolor="rgba(150,150,150,0.15)", layer="below", line_width=0)
        fig.add_vline(x=cut_date, line_width=3, line_dash="dash", line_color="black")
        fig.add_annotation(x=cut_date, y=max(I_real), text="Train → Prognose", showarrow=False,
                            yshift=20, font=dict(size=14, color="black"), bgcolor="white")

        fig.update_layout(
            title=f"{model_type} | β-Modus: {beta_mode} | RMSE={rmse:.1f}  MAE={mae:.1f}",
            yaxis_title="Infektionen",
            yaxis2=dict(title="R(t)", overlaying="y", side="right"),
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # ------------------------------------------------------------------
        # Ergebnis-Panel
        # ------------------------------------------------------------------
        beta_str = ", ".join(f"{b:.4f}" for b in beta_params)
        result_rows = [
            html.H6("Optimierungsergebnis", className="mt-2"),
            html.Div(f"N (effektive Population): {N_opt:,.0f}"),
            html.Div(f"I₀: {I0_opt:,.1f}"),
        ]
        if E0_opt is not None:
            result_rows.append(html.Div(f"E₀: {E0_opt:,.1f}"))
        result_rows += [
            html.Div(f"β-Parameter: [{beta_str}]"),
            html.Div(f"RMSE (Test): {rmse:,.2f}"),
            html.Div(f"MAE (Test): {mae:,.2f}"),
            html.Div(f"Laufzeit: {fit_seconds:.1f}s", className="text-muted"),
        ]

        return fig, result_rows

    # =====================================================
    # Callback für Modal
    # =====================================================
    @app.callback(
        Output("modal_info", "is_open"),
        Input("open_info", "n_clicks"),
        Input("close_info", "n_clicks"),
        State("modal_info", "is_open")
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    return app, server


def _build_fallback_app(error_text: str):
    """
    Minimal, dependency-light Flask app used ONLY if _build_app() above
    raised. Guarantees a `server` attribute always exists for gunicorn,
    and shows the real exception directly on the page so the problem is
    diagnosable from a browser without needing log access.
    """
    from flask import Flask

    fallback = Flask(__name__)

    @fallback.route("/")
    def _error_page():
        safe = error_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<html><head><title>Startup error</title></head><body "
            "style='font-family: monospace; background:#1e1e1e; color:#eee; padding:2em;'>"
            "<h2>App failed to start</h2>"
            "<p>The Dash app raised an exception during startup. Full traceback:</p>"
            f"<pre style='white-space: pre-wrap; background:#111; padding:1em; "
            f"border-radius:6px; overflow-x:auto;'>{safe}</pre>"
            "</body></html>",
            500,
        )

    return fallback


try:
    app, server = _build_app()
except Exception as _exc:  # noqa: BLE001 - last-resort guard, must not narrow
    BOOTSTRAP_ERROR = traceback.format_exc()
    print("FATAL: app failed to build during import:\n" + BOOTSTRAP_ERROR)
    app = None
    server = _build_fallback_app(BOOTSTRAP_ERROR)


# =====================================================
if __name__ == "__main__":
    if app is None:
        print(BOOTSTRAP_ERROR)
        raise SystemExit(1)
    url = "http://127.0.0.1:8050/"
    webbrowser.open(url)
    app.run(debug=True)