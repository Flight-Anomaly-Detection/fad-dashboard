import pandas as pd
import plotly.graph_objects as go
import pathlib
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
from dash.dependencies import Output, Input
import os


# DATA RETRIEVING FUNCTIONS
# No need for now.

# DATA PROCESSING FUNCTIONS
# No need for now.

# CHART-GENERATING FUNCTIONS
# Included @ callback section.

# SETTING UP PATH
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# GRABBING FLIGHT DATA
sample_flight = pd.read_parquet(DATA_PATH.joinpath('1a6bd1826bbed87ef815498494d52745.parquet'))

# LABELS AND SUCH
parameters_label = [{"label": i, "value": i} for i in sample_flight.columns.unique() if i != 'flight_id']
flights_label = [{"label": i.split('.')[0], "value": i.split('.')[0]} for i in os.listdir(DATA_PATH)]

# AUXILIARY DATA

# APP DEFINITION
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets')
server = app.server
app.config.suppress_callback_exceptions = True

# APP LAYOUT
NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Flight Data Anomaly Detection", className="ml-2"),
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="#",
        ),
    ],
    color="dark",
    dark=True,
    sticky="top",
)

PARAMETER_LINE_PLOT_SINGLE = [
    dbc.CardHeader(html.H5("Parameters chart - Single flight")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-single-flight",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-single-flight",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Select the a fight id and the flight parameters for the x and y axes:"),
                                    md=12),
                            dbc.Col(html.P("Flight id:"), md=6),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="flight-id",
                                    options=flights_label,
                                    value=flights_label[0]["value"],
                                ), ),
                            dbc.Col(
                                [
                                    dbc.Col(html.P("x-axis:"), md=6),
                                    dcc.Dropdown(
                                        id="parameter-x-single",
                                        options=parameters_label,
                                        value="Time",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Col(html.P("y-axis:"), md=6),
                                    dcc.Dropdown(
                                        id="parameter-y-single",
                                        options=parameters_label,
                                        value="Altitude",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="parameter-line-plot-single"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(PARAMETER_LINE_PLOT_SINGLE)), ], style={"marginTop": 30}),
    ],
    className="mt-12",
)

app.layout = html.Div(children=[NAVBAR,
                                BODY
                                ]
                      )


# CALLBACKS
@app.callback(
    Output("parameter-line-plot-single", "figure"),
    [Input("flight-id", "value"), Input("parameter-x-single", "value"), Input("parameter-y-single", "value")],
)
def gen_parameter_line_chart_single_flight(flight_id, parameter_x, parameter_y):
    df = pd.read_parquet(DATA_PATH.joinpath(flight_id+'.parquet'))
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=df[parameter_x].to_numpy(),
                             y=df[parameter_y].to_numpy(),
                             name=flight_id, mode='lines'))

    fig.update_xaxes(title_text=parameter_x)
    fig.update_yaxes(title_text=parameter_y)

    return fig


# Main
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080)
