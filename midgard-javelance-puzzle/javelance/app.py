"""Interactive Dash app for visualizing and selecting hexagons."""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback
from loguru import logger
from matplotlib import interactive

from javelance.javelance import JAVELANCE, JAVELANCE_GRID_SHAPE
from javelance.plotting import plot_shape_hexes, plot_shape
from javelance.shapes import Shape

HEX_SIZE = 1.0


def coords_to_hex(x, y, hex_size=1.0):
    """
    Convert click coordinates (x, y) to hex address (i, j).

    Uses the inverse of the hex coordinate transformation:
    - center = i * i_offset + j * j_offset
    - i_offset = [sqrt(3)*R, 0]
    - j_offset = [sqrt(3)/2*R, 3/2*R]
    """
    R = hex_size

    # Solve for j from y coordinate
    j = round(2 * y / (3 * R))

    # Solve for i from x coordinate
    i = round((x - j * np.sqrt(3) / 2 * R) / (np.sqrt(3) * R))

    return (i, j)


def create_figure(selected_nodes=None):
    """Create the main figure with JAVELANCE_GRID_SHAPE and JAVELANCE."""
    if selected_nodes is None:
        selected_nodes = set()

    logger.debug(f"Selected nodes:{selected_nodes}")

    # Use plot_shape_hexes to plot the grid with interactive hexagons
    fig = plot_shape_hexes(
        JAVELANCE_GRID_SHAPE, hex_size=HEX_SIZE, label_hexes=False, interactive=True
    )

    # Overlay JAVELANCE using plot_shape
    javelance_nodes = np.array(sorted(JAVELANCE.node_set()))
    javelance_edges = JAVELANCE.edges

    plot_shape(
        fig,
        javelance_nodes,
        javelance_edges,
        hex_size=HEX_SIZE,
        node_color=JAVELANCE.mean_color,
        edge_color=JAVELANCE.mean_color,
        alpha=0.5,
        inset_ratio=0.7,
    )

    # Overlay selected nodes using plot_shape
    if selected_nodes:
        selected_nodes_array = np.array(sorted(selected_nodes))
        selected_shape = Shape.from_sets(nodes=selected_nodes, edges=set())

        plot_shape(
            fig,
            selected_nodes_array,
            selected_shape.edges,
            hex_size=HEX_SIZE,
            node_color="orange",
            edge_color="orange",
            alpha=0.8,
            inset_ratio=0.85,
        )

    # Update title
    fig.update_layout(
        title="JAVELANCE Interactive Grid (Click hexagons to select/deselect)",
    )

    return fig


# Create the Dash app
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("JAVELANCE Interactive Hexagon Grid"),
        html.Div(
            [
                html.P("Click on hexagons to select them. Click again to deselect."),
                html.P(id="selection-info", children="No hexagons selected"),
            ]
        ),
        dcc.Graph(
            id="hex-grid", figure=create_figure(), config={"displayModeBar": True}
        ),
        dcc.Store(
            id="selected-nodes", data=[]
        ),  # Store selected nodes as list of [i, j]
    ]
)


@callback(
    Output("hex-grid", "figure"),
    Output("selected-nodes", "data"),
    Output("selection-info", "children"),
    Input("hex-grid", "clickData"),
    State("selected-nodes", "data"),
)
def handle_click(click_data, selected_nodes_data):
    """Handle clicks on hexagons to select/deselect them."""
    logger.debug("handle_click")
    # Convert stored data to set of tuples
    selected_nodes = (
        {tuple(node) for node in selected_nodes_data} if selected_nodes_data else set()
    )

    if click_data is not None and "points" in click_data:
        logger.debug(f"click_data: {click_data}")
        # Get the clicked point's custom data (hex address stored by plot_shape_hexes)
        point = click_data["points"][0]
        if "customdata" in point and point["customdata"] is not None:
            i, j = point["customdata"][0]

            logger.debug(f"Clicked hex address: ({i},{j})")

            # Validate that the clicked hex is in JAVELANCE_GRID_SHAPE
            if (i, j) in JAVELANCE_GRID_SHAPE.node_set():
                logger.debug(f"Clicked hex ({i},{j}) is in JAVELANCE_GRID_SHAPE")
                clicked_node = (i, j)

                # Toggle selection
                if clicked_node in selected_nodes:
                    selected_nodes.remove(clicked_node)
                else:
                    selected_nodes.add(clicked_node)
            else:
                logger.debug(f"Clicked hex ({i},{j}) is not in JAVELANCE_GRID_SHAPE")
        else:
            logger.debug(f"No customdata in point: {point.keys()}")
    else:
        logger.debug("No click data")

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
