import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import webbrowser

# =====================================================
# DATEN LADEN
# =====================================================
# Nutzt die kleine, reduzierte CSV direkt aus GitHub oder lokal
file_path = r"data/Aktuell_Deutschland_SarsCov2_Infektionen_total.csv"
df = pd.read_csv(file_path, parse_dates=["Meldedatum"])

# =====================================================
# SMOOTHING
# =====================================================
def smooth_series(data, window=7):
    return pd.Series(data).rolling(window, center=True, min_periods=1).mean().values

# =====================================================
# DYNAMISCHES BETA
# =====================================================
def beta_time(t, b0, b1, b2):
    return np.clip(b0 + b1*t + b2*t*t, 0, 2)

# =====================================================
# MODELLE
# =====================================================
def SIR(y, t, b0, b1, b2, gamma, N):
    S, I, R = y
    beta = beta_time(t, b0, b1, b2)
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def SEIR(y, t, b0, b1, b2, sigma, gamma, N):
    S, E, I, R = y
    beta = beta_time(t, b0, b1, b2)
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def SEIRS(y, t, b0, b1, b2, sigma, gamma, omega, N):
    S, E, I, R = y
    beta = beta_time(t, b0, b1, b2)
    dSdt = -beta * S * I / N + omega*R
    dEdt = beta * S * I / N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I - omega*R
    return [dSdt, dEdt, dIdt, dRdt]

# =====================================================
# PARAMETER FIT
# =====================================================
def fit_parameters(model_func, y0, t_train, I_train, N, sigma=0, gamma=1/10, omega=0):
    def loss(params):
        b0,b1,b2 = params

        if model_func == SIR:
            sol = odeint(model_func, y0, t_train, args=(b0,b1,b2,gamma,N))
            I_model = sol[:,1]
        elif model_func == SEIR:
            sol = odeint(model_func, y0, t_train, args=(b0,b1,b2,sigma,gamma,N))
            I_model = sol[:,2]
        else:
            sol = odeint(model_func, y0, t_train, args=(b0,b1,b2,sigma,gamma,omega,N))
            I_model = sol[:,2]

        scale = max(I_train)/(max(I_model)+1e-6)
        I_model *= scale

        return np.mean((I_model - I_train)**2)

    res = minimize(loss, x0=[0.4,0,0], bounds=[(0,2),(-0.05,0.05),(-0.001,0.001)])
    return res.x

# =====================================================
# DASH APP
# =====================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row(dbc.Col(html.H2("Dynamisches SEIRS Modell COVID (Deutschland gesamt)", className="text-center my-3"))),

    dbc.Row([

        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardBody([

                    html.Label("Zeitraum"),
                    dcc.DatePickerRange(
                        id="date_range",
                        start_date=df["Meldedatum"].min(),
                        end_date=df["Meldedatum"].max()
                    ),

                    html.Br(),
                    html.Label("Modelltyp"),
                    dcc.RadioItems(
                        id="model_type",
                        options=["SIR","SEIR","SEIRS"],
                        value="SEIR",
                        inline=True
                    ),

                    html.Br(),
                    html.Label("Fit Datenbasis"),
                    dcc.RadioItems(
                        id="fit_mode",
                        options=[
                            {"label":"Rohdaten","value":"raw"},
                            {"label":"7-Tage-Mittel","value":"smooth"}
                        ],
                        value="smooth",
                        inline=True
                    ),

                    html.Br(),
                    html.Label("Population"),
                    dcc.Slider(id="population",min=0,max=1000000,step=5000,value=83000000), # Deutschland gesamt

                    html.Br(),
                    html.Label("I₀"),
                    dcc.Slider(id="I0",min=0,max=10000,step=10,value=100),

                    html.Br(),
                    html.Label("Train Split (%)"),
                    dcc.Slider(id="split_ratio",min=10,max=90,step=5,value=70),

                ])
            ])
        ], width=3),

        # Plot
        dbc.Col(dcc.Graph(id="simulation_plot"), width=9)

    ])
])

# =====================================================
# CALLBACK
# =====================================================
@app.callback(
    Output("simulation_plot","figure"),
    Input("date_range","start_date"),
    Input("date_range","end_date"),
    Input("model_type","value"),
    Input("population","value"),
    Input("I0","value"),
    Input("split_ratio","value"),
    Input("fit_mode","value")
)
def update_simulation(start,end,model_type,N,I0,split_ratio,fit_mode):

    # Daten filtern (nur nach Datum)
    df_period = df[(df["Meldedatum"]>=start) & (df["Meldedatum"]<=end)]
    daily = df_period.groupby("Meldedatum")["AnzahlFall"].sum().sort_index()
    dates = daily.index
    I_real = daily.values

    if len(I_real)<30:
        return go.Figure()

    # 7-Tage-Mittel immer berechnen
    I_smooth = smooth_series(I_real,7)

    # Fit-Daten bestimmen (Rohdaten oder geglättet)
    I_used = I_smooth if fit_mode=="smooth" else I_real.copy()

    # Daten aufteilen
    split_idx = int(len(I_real)*split_ratio/100)
    I_train = I_used[:split_idx]
    I_test  = I_real[split_idx:]

    t_train = np.arange(len(I_train))
    t_total = np.arange(len(I_real))

    gamma, sigma, omega = 1/10, 1/5, 1/180
    E0, R0 = 2*I0, 0
    S0 = N-I0-E0

    model_map = {"SIR":SIR,"SEIR":SEIR,"SEIRS":SEIRS}
    y0 = [S0,I0,R0] if model_type=="SIR" else [S0,E0,I0,R0]
    model_func = model_map[model_type]

    b0,b1,b2 = fit_parameters(model_func,y0,t_train,I_train,N,sigma,gamma,omega)

    # simulate
    if model_type=="SIR":
        sol = odeint(SIR,y0,t_total,args=(b0,b1,b2,gamma,N))
        I_model = sol[:,1]
    else:
        sol = odeint(model_func,y0,t_total,args=(b0,b1,b2,sigma,gamma,N) if model_type=="SEIR"
                     else (b0,b1,b2,sigma,gamma,omega,N))
        I_model = sol[:,2]

    scale = max(I_train)/(max(I_model[:split_idx])+1e-6)
    I_model *= scale

    rmse = np.sqrt(mean_squared_error(I_test,I_model[split_idx:]))
    mae = mean_absolute_error(I_test,I_model[split_idx:])

    # R(t)
    R_t = beta_time(t_total,b0,b1,b2)/gamma

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=I_real,mode="markers",name="Rohdaten",opacity=0.4))
    fig.add_trace(go.Scatter(x=dates, y=I_smooth, mode="lines", name="7-Tage-Mittel"))
    fig.add_trace(go.Scatter(
        x=dates,
        y=I_model,
        mode="lines",
        name="Modell",
        line=dict(color="red", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=R_t,
        mode="lines",
        name="R(t)",
        yaxis="y2",
        line=dict(color="green", width=3)
    ))

    cut_date = dates[split_idx]

    # Grauer Prognosebereich
    fig.add_vrect(
        x0=cut_date,
        x1=dates[-1],
        fillcolor="rgba(150,150,150,0.15)",
        layer="below",
        line_width=0,
    )

    # Vertikale Trennlinie
    fig.add_vline(
        x=cut_date,
        line_width=3,
        line_dash="dash",
        line_color="black"
    )

    # Beschriftungen
    fig.add_annotation(
        x=cut_date,
        y=max(I_real),
        text="Train → Prognose",
        showarrow=False,
        yshift=20,
        font=dict(size=14,color="black"),
        bgcolor="white"
    )

    fig.update_layout(
        title=f"{model_type} | RMSE={rmse:.1f} MAE={mae:.1f}",
        yaxis_title="Infektionen",
        yaxis2=dict(title="R(t)",overlaying="y",side="right"),
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

# =====================================================
if __name__=="__main__":
    url="http://127.0.0.1:8050/"
    webbrowser.open(url)
    app.run(debug=True)
