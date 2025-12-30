"""Interactive Dash app for visualizing and selecting hexagons."""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback, Patch
from loguru import logger

from javelance.javelance import (
    JAVELANCE,
    JAVELANCE_GRID_SHAPE,
    JAVELANCE_REGIONS,
    JAVELANCE_FORBIDDEN,
)
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


def create_base_figure():
    """Create the base figure with JAVELANCE_GRID_SHAPE and JAVELANCE (no selected nodes)."""
    # Use plot_shape_hexes to plot the grid with interactive hexagons
    fig = plot_shape_hexes(
        JAVELANCE_GRID_SHAPE, hex_size=HEX_SIZE, label_hexes=False, interactive=True
    )

    for region_id, region in JAVELANCE_REGIONS.items():
        region.plot(
            fig, hex_size=HEX_SIZE, interactive=True, labels=lambda i, j: region_id
        )

    JAVELANCE_FORBIDDEN.plot(fig, hex_size=HEX_SIZE, alpha=0.5)

    # Update title
    fig.update_layout(
        title="JAVELANCE Interactive Grid (Click hexagons to select/deselect)",
    )

    return fig


def add_selected_nodes_to_patch(patch, selected_nodes, start_index):
    """
    Add selected node traces to a Patch object.

    Args:
        patch: Patch object to add traces to
        selected_nodes: Set of selected node tuples (i, j)
        start_index: Index where the first trace will be added

    Returns:
        List of indices where traces were added
    """
    if not selected_nodes:
        return []

    selected_nodes_array = np.array(sorted(selected_nodes))
    selected_shape = Shape.from_sets(nodes=selected_nodes, edges=set())

    # Create a minimal temporary figure to get just the selected node traces
    import plotly.graph_objects as go

    temp_fig = go.Figure()

    plot_shape(
        temp_fig,
        selected_nodes_array,
        selected_shape.edges,
        hex_size=HEX_SIZE,
        node_color="orange",
        edge_color="orange",
        alpha=0.8,
        inset_ratio=0.85,
        interactive=True,  # Make selected hexagons clickable to allow deselection
    )

    # Add all traces from the temporary figure to the patch and track indices
    added_indices = []
    for i, trace in enumerate(temp_fig.data):
        patch.data.append(trace)
        added_indices.append(start_index + i)

    return added_indices


# Create the Dash app
app = Dash(__name__)

# Create the base figure once at startup
BASE_FIGURE = create_base_figure()
NUM_BASE_TRACES = len(BASE_FIGURE.data)

app.layout = html.Div(
    [
        html.H1("JAVELANCE Interactive Hexagon Grid"),
        html.Div(
            [
                html.P("Click on hexagons to select them. Click again to deselect."),
                html.P(id="selection-info", children="No hexagons selected"),
            ]
        ),
        dcc.Graph(id="hex-grid", figure=BASE_FIGURE, config={"displayModeBar": True}),
        dcc.Store(
            id="selected-nodes", data=[]
        ),  # Store selected nodes as list of [i, j]
        dcc.Store(
            id="selected-trace-indices", data=[]
        ),  # Track indices of traces added for selected nodes
    ]
)


@callback(
    Output("hex-grid", "figure"),
    Output("selected-nodes", "data"),
    Output("selection-info", "children"),
    Output("selected-trace-indices", "data"),
    Input("hex-grid", "clickData"),
    State("selected-nodes", "data"),
    State("selected-trace-indices", "data"),
)
def handle_click(click_data, selected_nodes_data, selected_trace_indices):
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

    # Use Patch to efficiently update only the selected nodes traces
    patched_figure = Patch()

    # Remove traces at the exact indices from the previous selection
    # Delete in reverse order (largest index first) to avoid index shifting
    if selected_trace_indices is None:
        selected_trace_indices = []

    logger.debug(
        f"Removing {len(selected_trace_indices)} previous selected node traces at indices: {selected_trace_indices}"
    )
    for idx in sorted(selected_trace_indices, reverse=True):
        try:
            del patched_figure.data[idx]
        except (IndexError, KeyError):
            logger.warning(f"Failed to delete trace at index {idx}")

    # Add new selected node traces starting at NUM_BASE_TRACES
    # (since we've deleted all the old selected traces, new ones start right after base)
    new_trace_indices = add_selected_nodes_to_patch(
        patched_figure, selected_nodes, NUM_BASE_TRACES
    )
    logger.debug(
        f"Added {len(new_trace_indices)} new selected node traces at indices: {new_trace_indices}"
    )

    # Convert set back to list for storage
    selected_nodes_list = [list(node) for node in selected_nodes]

    # Create info message
    if selected_nodes:
        info = f"Selected {len(selected_nodes)} hexagon(s): {sorted(selected_nodes)}"
    else:
        info = "No hexagons selected"

    return patched_figure, selected_nodes_list, info, new_trace_indices


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
