import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import webbrowser

# =====================================================
# BOOTSTRAP GUARD
# =====================================================
# Everything that builds the real app (model import, Dash construction,
# layout, callbacks) happens inside _build_app(). If ANY of that throws —
# for any reason, on any host — we still end up with a working `server`
# attribute, because WSGI hosts (gunicorn etc.) look that up by name right
# after importing this file. Without this guard, an exception anywhere in
# that chain prevents `server` from ever being defined, and the host reports
# a confusing "module 'app' has no attribute 'server'" instead of the real
# cause. With this guard, the real traceback is printed to the logs AND
# rendered directly on the page, so it's diagnosable without redeploying.

BOOTSTRAP_ERROR = None


def _build_app():
    """Build and return (app, server). May raise — caller handles fallback."""
    import covid_model_local_Experimental_Beta_Relative_differentialEvolution as cm

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    server = app.server

    # -------------------------------------------------
    # DATEN LADEN
    # -------------------------------------------------
    data_load_error = None
    try:
        df = cm.load_data(cm.FILE_PATH)
    except Exception as exc:  # noqa: BLE001 - intentionally broad: surface in UI, not just logs
        data_load_error = f"{exc}"
        print("ERROR loading RKI data:\n" + traceback.format_exc())
        df = pd.DataFrame({"Meldedatum": pd.to_datetime([]), "AnzahlFall": []})

    BETA_MODE_OPTIONS = [
        {"label": "Konstant",   "value": "constant"},
        {"label": "Linear",     "value": "linear"},
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
        I_smooth = cm.smooth_series(I_real, 7)
        I_used = I_smooth if fit_mode == "smooth" else I_real.copy()

        # Split Train/Test
        split_idx = int(len(I_real) * split_ratio / 100)
        split_idx = max(2, min(split_idx, len(I_real) - 1))
        I_train = I_used[:split_idx]
        I_test = I_real[split_idx:]

        t_train = np.arange(len(I_train), dtype=float)
        t_total = np.arange(len(I_real), dtype=float)

        model_func = cm.MODEL_MAP[model_type]

        # ------------------------------------------------------------------
        # Globale Optimierung (differential_evolution) — Methodik aus
        # covid_model_local_Experimental_Beta_Relative_differentialEvolution.py
        # ------------------------------------------------------------------
        t0 = time.time()
        beta_params, N_opt, I0_opt, E0_opt = cm.fit_parameters(
            model_func, t_train, I_train, beta_mode,
            sigma=cm.SIGMA, gamma=cm.GAMMA, omega=cm.OMEGA,
            n_min=cm.N_MIN, n_max=cm.N_MAX,
            de_maxiter=de_maxiter, de_popsize=de_popsize,
            de_tol=cm.DE_TOL, de_seed=cm.DE_SEED,
            de_workers=1, de_polish=cm.DE_POLISH,
        )
        fit_seconds = time.time() - t0

        # ------------------------------------------------------------------
        # Vollständige Simulation mit optimierten Parametern
        # ------------------------------------------------------------------
        y0 = cm.build_initial_conditions(model_func, N_opt, I0_opt, E0_opt, R0_init=0)
        args = cm.build_args(model_func, beta_params, beta_mode, cm.SIGMA, cm.GAMMA, cm.OMEGA, N_opt)
        sol = cm.run_model(model_func, y0, t_total, args, tight=True)
        I_model = cm.extract_I(model_func, sol)

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
        beta_t = cm.beta_time(t_total, beta_params, beta_mode)
        R_t = beta_t / cm.GAMMA * (S_t / N_opt)

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

