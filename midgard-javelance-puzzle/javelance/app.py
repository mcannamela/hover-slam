"""Interactive Dash app for visualizing and selecting hexagons."""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

from javelance.javelance import JAVELANCE, JAVELANCE_GRID_SHAPE


def create_hex_trace(i, j, hex_size=1.0, color="black", fill_color=None, opacity=1.0, customdata=None):
    """Create a plotly trace for a single hexagon."""
    R = hex_size
    i_offset = np.array([np.sqrt(3) * R, 0])
    j_offset = np.array([np.sqrt(3) / 2 * R, 3 / 2 * R])

    # Calculate center position
    center = i * i_offset + j * j_offset
    cx, cy = center

    # Vertices of a pointy-top hexagon
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    vertices_x = cx + R * np.cos(angles)
    vertices_y = cy + R * np.sin(angles)

    # Close the hexagon
    vertices_x = np.append(vertices_x, vertices_x[0])
    vertices_y = np.append(vertices_y, vertices_y[0])

    trace_config = {
        "x": vertices_x,
        "y": vertices_y,
        "mode": "lines",
        "line": dict(color=color, width=1),
        "showlegend": False,
        "hoverinfo": "text",
        "hovertext": f"({i}, {j})",
        "customdata": [[i, j]] * len(vertices_x),  # Store node coordinates
    }

    if fill_color is not None:
        trace_config["fill"] = "toself"
        trace_config["fillcolor"] = fill_color
        trace_config["opacity"] = opacity

    return go.Scatter(**trace_config)


def create_figure(selected_nodes=None):
    """Create the main figure with JAVELANCE_GRID_SHAPE and JAVELANCE."""
    if selected_nodes is None:
        selected_nodes = set()

    fig = go.Figure()

    # Get nodes from JAVELANCE_GRID_SHAPE
    grid_nodes = JAVELANCE_GRID_SHAPE.node_set()
    javelance_nodes = JAVELANCE.node_set()

    # Plot grid hexagons
    for node in grid_nodes:
        i, j = node

        # Determine color based on whether it's in JAVELANCE and/or selected
        if node in selected_nodes:
            # Selected nodes - bright highlight
            color = "orange"
            fill_color = "rgba(255, 165, 0, 0.6)"
        elif node in javelance_nodes:
            # JAVELANCE nodes - light fill
            color = "black"
            fill_color = "rgba(245, 222, 179, 0.5)"  # Bisque with transparency
        else:
            # Grid-only nodes - no fill
            color = "lightgray"
            fill_color = None

        trace = create_hex_trace(i, j, hex_size=1.0, color=color, fill_color=fill_color)
        fig.add_trace(trace)

    # Calculate bounds for aspect ratio
    if len(grid_nodes) > 0:
        min_addr, max_addr = JAVELANCE_GRID_SHAPE.bounding_addresses()
        R = 1.0
        max_x = (max_addr[0] - min_addr[0]) * np.sqrt(3) * R + (max_addr[1] - min_addr[1]) * np.sqrt(3) / 2 * R + 4 * R
        max_y = (max_addr[1] - min_addr[1]) * 3 / 2 * R + 4 * R
    else:
        max_x = 4
        max_y = 4

    # Set a base height and calculate width to match the aspect ratio
    base_height = 800
    aspect_ratio = max_x / max_y if max_y > 0 else 1.0
    plot_width = int(base_height * aspect_ratio)
    plot_height = base_height

    # Set equal aspect ratio and clean layout
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="JAVELANCE Interactive Grid (Click hexagons to select/deselect)",
    )

    return fig


# Create the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("JAVELANCE Interactive Hexagon Grid"),
    html.Div([
        html.P("Click on hexagons to select them. Click again to deselect."),
        html.P(id="selection-info", children="No hexagons selected"),
    ]),
    dcc.Graph(id="hex-grid", figure=create_figure(), config={"displayModeBar": True}),
    dcc.Store(id="selected-nodes", data=[]),  # Store selected nodes as list of [i, j]
])


@callback(
    Output("hex-grid", "figure"),
    Output("selected-nodes", "data"),
    Output("selection-info", "children"),
    Input("hex-grid", "clickData"),
    State("selected-nodes", "data"),
)
def handle_click(click_data, selected_nodes_data):
    """Handle clicks on hexagons to select/deselect them."""
    # Convert stored data to set of tuples
    selected_nodes = {tuple(node) for node in selected_nodes_data} if selected_nodes_data else set()

    if click_data is not None and "points" in click_data:
        # Get the clicked point's custom data
        point = click_data["points"][0]
        if "customdata" in point and point["customdata"] is not None:
            i, j = point["customdata"]
            clicked_node = (i, j)

            # Toggle selection
            if clicked_node in selected_nodes:
                selected_nodes.remove(clicked_node)
            else:
                selected_nodes.add(clicked_node)

    # Create updated figure
    fig = create_figure(selected_nodes)

    # Convert set back to list for storage
    selected_nodes_list = [list(node) for node in selected_nodes]

    # Create info message
    if selected_nodes:
        info = f"Selected {len(selected_nodes)} hexagon(s): {sorted(selected_nodes)}"
    else:
        info = "No hexagons selected"

    return fig, selected_nodes_list, info


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
