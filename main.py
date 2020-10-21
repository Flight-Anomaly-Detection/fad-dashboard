import numpy as np
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# DATA RETRIEVING FUNCTIONS
# No need for now.

# DATA PROCESSING FUNCTIONS
# No need for now.

# LAYOUT CREATING FUNCTIONS
def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


# CHART-GENERATING FUNCTIONS
# Included @ callback section.
def generate_figure_image(groups, layout):
    data = []
    for idx, val in groups:
        scatter = go.Scatter3d(
            name=idx,
            x=val["x"],
            y=val["y"],
            z=val["z"],
            text=val['flight_id'],
            textposition="top center",
            mode="markers",
            marker=dict(size=3, symbol="circle"),
        )
        data.append(scatter)

    figure = go.Figure(data=data, layout=layout)

    return figure


# SETTING UP PATH
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
FLIGHTS_PATH = DATA_PATH.joinpath("flights").resolve()

# GRABBING DATA
sample_flight = pd.read_parquet(FLIGHTS_PATH.joinpath('1a6bd1826bbed87ef815498494d52745.parquet'))
X_orig = np.load(DATA_PATH.joinpath('X_array.npy'))
df_flights = pd.read_csv(DATA_PATH.joinpath('df_flights.csv'))

# LABELS AND SUCH
parameters_label = [{"label": i, "value": i} for i in sample_flight.columns.unique() if i != 'flight_id']
flights_label = [{"label": i.split('.')[0], "value": i.split('.')[0]} for i in os.listdir(FLIGHTS_PATH)]

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

CLUSTERING = [
    dbc.CardHeader(html.H5("Clustering")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-tsne",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-tsne",
                        color="warning",
                        style={"display": "none"},
                    ),
                    html.Label("PCA parameters:", className="lead"),
                    NamedSlider(
                        name="Initial PCA Dimensions",
                        short="pca-dimension",
                        min=25,
                        max=100,
                        step=None,
                        val=50,
                        marks={i: str(i) for i in [25, 50, 75, 100]},
                    ),
                    html.Label("DBSCAN parameters:", className="lead"),
                    NamedSlider(
                        name="eps",
                        short="eps",
                        min=10,
                        max=200,
                        step=None,
                        val=100,
                        marks={i: str(i) for i in range(10,200,10)},
                    ),
                    html.Label("t-SNE parameters:", className="lead"),
                    NamedSlider(
                        name="Number Of Iterations",
                        short="iterations",
                        min=250,
                        max=1000,
                        step=None,
                        val=500,
                        marks={
                            i: str(i) for i in [250, 500, 750, 1000]
                        },
                    ),
                    NamedSlider(
                        name="Perplexity",
                        short="perplexity",
                        min=3,
                        max=100,
                        step=None,
                        val=30,
                        marks={i: str(i) for i in [3, 10, 30, 50, 100]},
                    ),
                    NamedSlider(
                        name="Learning Rate",
                        short="learning-rate",
                        min=10,
                        max=200,
                        step=None,
                        val=100,
                        marks={i: str(i) for i in [10, 50, 100, 200]},
                    ),
                    html.Label("t-SNE 3D plot", className="lead"),
                    dcc.Graph(id="graph-3d-plot-tsne"),
                    html.Label("Clustering metrics", className="lead"),
                    dbc.Col(id='silhouette-score', children="Teste", md=12)
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

PRINCIPAL_COMPONENT_ANALYSIS = []

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(PARAMETER_LINE_PLOT_SINGLE)), ], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CLUSTERING)), ], style={"marginTop": 30}),
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
    df = pd.read_parquet(FLIGHTS_PATH.joinpath(flight_id+'.parquet'))
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=df[parameter_x].to_numpy(),
                             y=df[parameter_y].to_numpy(),
                             name=flight_id, mode='lines'))

    fig.update_xaxes(title_text=parameter_x)
    fig.update_yaxes(title_text=parameter_y)

    return fig


@app.callback(
    [
        Output("graph-3d-plot-tsne", "figure"),
        Output("silhouette-score", "children"),
    ],
    [
        Input("flight-id", "value"),
        Input("slider-iterations", "value"),
        Input("slider-perplexity", "value"),
        Input("slider-pca-dimension", "value"),
        Input("slider-learning-rate", "value"),
        Input("slider-eps", "value"),
    ],
)
def display_3d_scatter_plot(
    useless,
    iterations,
    perplexity,
    pca_dim,
    learning_rate,
    eps
):
    try:
        pipeline_steps = [('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_dim)),
                ('dbscan', DBSCAN(eps=eps, min_samples=1))]
        pipeline = Pipeline(pipeline_steps)
        db_scaled = pipeline.fit(X_orig)
        labels = pipeline['dbscan'].labels_

        # Grabbing pca vector
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_orig)
        pca = PCA(n_components=pca_dim)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        
        #Tsne for 3d
        model = TSNE()
        tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, n_iter=iterations, n_jobs=5).fit_transform(X_pca)
        embedding_df = pd.DataFrame(tsne, columns=['x','y','z'])
        embedding_df.set_index(labels, inplace=True)

        # Plot layout
        axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )

        embedding_df["label"] = embedding_df.index
        embedding_df["flight_id"] = df_flights['flight_id']
        groups = embedding_df.groupby("label")
        figure = generate_figure_image(groups, layout)

        try:
            silhouette_avg = silhouette_score(X_pca, labels)
        except:
            silhouette_avg = np.nan
            sample_silhouette_values = []

        if silhouette_avg is not np.nan:
            silhouette_children = 'Average silhouette: {}'.format(round(silhouette_avg, 3))
        else:
            silhouette_children = "Not applicable."

    except:
        figure = go.Figure()
        silhouette_children = "Not applicable." 

    return (figure, silhouette_children)

# Main
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080)
