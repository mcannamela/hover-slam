import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.graph_objs import Figure

from javelance.packing import PackingSolution
from javelance.shapes import Shape, union_shapes
from javelance.javelance import (
    JAVELANCE_GRID_SHAPE,
    JAVELANCE,
    JAVELANCE_FORBIDDEN,
    JAVELANCE_REGIONS,
)

# Set default renderer to always open plots in browser
pio.renderers.default = "browser"


def plot_shape_hexes(shape, hex_size=1.0, label_hexes=True, interactive=False):
    """
    Plot hexagons for all nodes in a shape, with interior and boundary edges.

    Parameters:
    - shape: Shape object containing nodes to plot
    - hex_size: circumradius of each hexagon (distance from center to vertex)
    - label_hexes: if True, label every 5th hex with its (i, j) coordinates
    - interactive: if True, make hexagons clickable (don't skip hover info)

    Returns:
    - Plotly Figure object
    """
    # For pointy-top hexagons with circumradius R:
    # - i offset vector: (sqrt(3) * R, 0)
    # - j offset vector: (sqrt(3)/2 * R, 3/2 * R)
    R = hex_size
    i_offset = np.array([np.sqrt(3) * R, 0])
    j_offset = np.array([np.sqrt(3) / 2 * R, 3 / 2 * R])

    # Vertices of a pointy-top hexagon are at angles: 30°, 90°, 150°, 210°, 270°, 330°
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180

    fig = go.Figure()

    # Get nodes and edges
    nodes = shape.node_set()
    interior_edges = shape.interior_edges()
    boundary_edges = shape.boundary_edges()

    # Plot all hexagons for nodes in the shape
    for node in nodes:
        i, j = node
        # Calculate center position of hexagon (i, j)
        center = i * i_offset + j * j_offset
        cx, cy = center

        # Calculate vertices
        vertices_x = cx + R * np.cos(angles)
        vertices_y = cy + R * np.sin(angles)

        # Close the hexagon by adding the first vertex at the end
        vertices_x = np.append(vertices_x, vertices_x[0])
        vertices_y = np.append(vertices_y, vertices_y[0])

        # Add hexagon outline (and fill if interactive)
        trace_params = {
            "x": vertices_x,
            "y": vertices_y,
            "mode": "lines",
            "line": dict(color="black", width=1),
            "showlegend": False,
        }

        if interactive:
            # Make filled and clickable
            trace_params["fill"] = "toself"
            trace_params["fillcolor"] = (
                "rgba(255, 255, 255, 0.01)"  # Nearly transparent
            )
            trace_params["hoverinfo"] = (
                "none"  # Don't show hover text, but allow clicks
            )
            # Store hex address in customdata so clicks can identify which hex was clicked
            trace_params["customdata"] = [[i, j]] * len(vertices_x)
        else:
            # Non-interactive outline only
            trace_params["hoverinfo"] = "skip"

        fig.add_trace(go.Scatter(**trace_params))

    # Add labels to every 5th hex
    if label_hexes:
        for node in nodes:
            i, j = node
            # Label every 5th hex
            if (i + j) % 5 == 0:
                center = i * i_offset + j * j_offset
                cx, cy = center

                fig.add_annotation(
                    x=cx,
                    y=cy,
                    text=f"{i}, {j}",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor="center",
                    yanchor="middle",
                )

    # Calculate bounds for aspect ratio
    if len(nodes) > 0:
        min_addr, max_addr = shape.bounding_addresses()
        # Maximum extent in x and y
        max_x = (
            (max_addr[0] - min_addr[0]) * np.sqrt(3) * R
            + (max_addr[1] - min_addr[1]) * np.sqrt(3) / 2 * R
            + 4 * R
        )
        max_y = (max_addr[1] - min_addr[1]) * 3 / 2 * R + 4 * R
    else:
        max_x = 4 * R
        max_y = 4 * R

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
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


def plot_hex_grid(
    width: int, height: int, hex_size=1.0, exclude=None, label_hexes=True
):
    """
    Plot a hexagonal grid with pointy-top orientation.

    Parameters:
    - I: number of hexagons in the i direction (aligned with x-axis)
    - J: number of hexagons in the j direction (60° counterclockwise from x-axis)
    - hex_size: circumradius of each hexagon (distance from center to vertex)
    - exclude: nx2 array of (i, j) pairs to exclude from plotting (optional)
    - label_hexes: if True, label every 5th hex with its (i, j) coordinates

    The hexagons are indexed by (i, j) where:
    - i direction is aligned with the x-axis
    - j direction is 60° counterclockwise from the x-axis
    """
    # Create a box shape for the grid
    grid_shape = Shape.box(width=width, height=height)

    # If there are nodes to exclude, remove them from the grid shape
    if exclude is not None:
        exclude_nodes = Shape.as_node_set(exclude)
        grid_shape = grid_shape.difference(
            Shape.from_sets(nodes=exclude_nodes, edges=set())
        )

    # Use plot_shape_hexes to plot the grid
    return plot_shape_hexes(grid_shape, hex_size=hex_size, label_hexes=label_hexes)


def _hex_center(i, j, hex_size=1.0):
    """Calculate the center position of a hexagon at address (i, j)."""
    R = hex_size
    i_offset = np.array([np.sqrt(3) * R, 0])
    j_offset = np.array([np.sqrt(3) / 2 * R, 3 / 2 * R])
    return i * i_offset + j * j_offset


def _shared_edge_vertices(hex1, hex2, hex_size=1.0):
    """
    Calculate the vertices of the shared edge between two adjacent hexagons.

    Parameters:
    - hex1: (i, j) address of first hexagon
    - hex2: (i, j) address of second hexagon
    - hex_size: circumradius of hexagons

    Returns:
    - Two points defining the shared edge: [(x1, y1), (x2, y2)]
    """
    R = hex_size
    i1, j1 = hex1
    i2, j2 = hex2

    # Calculate the offset
    di = i2 - i1
    dj = j2 - j1

    # Map offset direction to the pair of vertex angles that define the shared edge
    # For pointy-top hexagons, vertices are at: 30°, 90°, 150°, 210°, 270°, 330°
    offset_to_vertices = {
        (1, 0): (330, 30),  # right edge
        (-1, 0): (150, 210),  # left edge
        (0, 1): (30, 90),  # upper-right edge
        (0, -1): (210, 270),  # lower-left edge
        (1, -1): (270, 330),  # lower-right edge
        (-1, 1): (90, 150),  # upper-left edge
    }

    offset_key = (di, dj)
    if offset_key not in offset_to_vertices:
        raise ValueError(f"Hexagons {hex1} and {hex2} are not adjacent")

    vertex_angles = offset_to_vertices[offset_key]

    # Get the center of the first hexagon
    center = _hex_center(i1, j1, hex_size)

    # Calculate the two vertices of the shared edge
    v1_angle = vertex_angles[0] * np.pi / 180
    v2_angle = vertex_angles[1] * np.pi / 180

    v1 = center + R * np.array([np.cos(v1_angle), np.sin(v1_angle)])
    v2 = center + R * np.array([np.cos(v2_angle), np.sin(v2_angle)])

    return v1, v2


def plot_shape(
    fig,
    nodes,
    edges,
    hex_size=1.0,
    node_color="red",
    edge_color="blue",
    jitter=0.1,
    alpha=0.7,
    inset_ratio=0.7,
    interactive=False,
    labels=None,
):
    """
    Plot a shape on the hexagonal grid.

    Parameters:
    - fig: plotly Figure object to add traces to
    - nodes: Nx2 array of hex addresses (i, j)
    - edges: Mx2x2 array where each edge is [[i1, j1], [i2, j2]]
    - hex_size: circumradius of hexagons
    - node_color: color for node markers
    - edge_color: color for edge lines
    - jitter: amount to offset edges inward (as fraction of hex_size)
    - alpha: transparency for nodes and edges (0-1)
    - inset_ratio: ratio of node hexagon size to grid hexagon size
    - interactive: if True, make hexagons clickable with customdata
    - labels: optional labels for hexagons. Can be:
        - None: no labels (default)
        - dict: mapping (i, j) tuples to label strings
        - callable: function taking (i, j) and returning label string
        - list/array: parallel to nodes, one label per node
    """
    # Vertices of a pointy-top hexagon are at angles: 30°, 90°, 150°, 210°, 270°, 330°
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180

    # Plot nodes as filled hexagons
    if len(nodes) > 0:
        for node in nodes:
            center = _hex_center(node[0], node[1], hex_size)
            cx, cy = center

            # Calculate inset hexagon vertices
            R_inset = hex_size * inset_ratio
            vertices_x = cx + R_inset * np.cos(angles)
            vertices_y = cy + R_inset * np.sin(angles)

            # Close the hexagon
            vertices_x = np.append(vertices_x, vertices_x[0])
            vertices_y = np.append(vertices_y, vertices_y[0])

            # Convert color to rgba format with alpha
            if node_color.startswith("rgb"):
                # Already in rgb format, convert to rgba
                rgba_color = node_color.replace("rgb", "rgba").replace(
                    ")", f", {alpha})"
                )
            else:
                # Named color, use directly with opacity parameter
                rgba_color = node_color

            trace_params = {
                "x": vertices_x,
                "y": vertices_y,
                "mode": "lines",
                "line": dict(color=rgba_color, width=1),
                "fill": "toself",
                "fillcolor": rgba_color,
                "opacity": alpha,
                "showlegend": False,
            }

            if interactive:
                # Make clickable with customdata
                trace_params["hoverinfo"] = "none"
                trace_params["customdata"] = [[node[0], node[1]]] * len(vertices_x)
            else:
                trace_params["hoverinfo"] = "skip"

            fig.add_trace(go.Scatter(**trace_params))

    # Plot edges
    for edge in edges:
        hex1 = tuple(edge[0])
        hex2 = tuple(edge[1])

        # Get the shared edge vertices
        v1, v2 = _shared_edge_vertices(hex1, hex2, hex_size)

        # Calculate edge vector
        edge_vec = v2 - v1

        # Calculate perpendicular to edge (rotate 90 degrees)
        # Two perpendiculars: (-dy, dx) and (dy, -dx)
        perp1 = np.array([-edge_vec[1], edge_vec[0]])

        # Normalize the perpendicular
        perp1 = perp1 / np.linalg.norm(perp1)

        # Determine which perpendicular points toward the interior
        # (toward the midpoint between the two hex centers)
        edge_center = (v1 + v2) / 2
        center1 = _hex_center(hex1[0], hex1[1], hex_size)
        center2 = _hex_center(hex2[0], hex2[1], hex_size)
        midpoint = (center1 + center2) / 2

        # Choose the perpendicular direction that points toward the midpoint
        to_midpoint = midpoint - edge_center
        if np.dot(perp1, to_midpoint) < 0:
            perp1 = -perp1

        # Apply jitter orthogonal to the edge
        jitter_vec = perp1 * jitter * hex_size

        # Apply jitter to edge vertices
        v1_jittered = v1 + jitter_vec
        v2_jittered = v2 + jitter_vec

        # Convert edge color to rgba format with alpha
        if edge_color.startswith("rgb"):
            # Already in rgb format, convert to rgba
            edge_rgba_color = edge_color.replace("rgb", "rgba").replace(
                ")", f", {alpha})"
            )
        else:
            # Named color, use directly with opacity parameter
            edge_rgba_color = edge_color

        # Plot the edge
        fig.add_trace(
            go.Scatter(
                x=[v1_jittered[0], v2_jittered[0]],
                y=[v1_jittered[1], v2_jittered[1]],
                mode="lines",
                line=dict(color=edge_rgba_color, width=3),
                opacity=alpha,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add labels to hexagon centers if requested
    if labels is not None and len(nodes) > 0:
        for idx, node in enumerate(nodes):
            i, j = node[0], node[1]
            center = _hex_center(i, j, hex_size)
            cx, cy = center

            # Determine the label text for this node
            label_text = None
            if callable(labels):
                # labels is a function: call it with (i, j)
                label_text = labels(i, j)
            elif isinstance(labels, dict):
                # labels is a dict: look up (i, j)
                label_text = labels.get((i, j))
            elif hasattr(labels, "__getitem__"):
                # labels is a list/array: use index
                if idx < len(labels):
                    label_text = labels[idx]

            # Add annotation if we have a label
            if label_text is not None:
                fig.add_annotation(
                    x=cx,
                    y=cy,
                    text=str(label_text),
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xanchor="center",
                    yanchor="middle",
                )

    return fig


def plot_boundary_edges(
    fig,
    shape,
    hex_size=1.0,
    color=None,
    alpha=1.0,
    width=5,
    jitter_amount=30,
):
    """
    Plot the boundary edges of a shape with thick lines.

    Parameters:
    - fig: plotly Figure object to add traces to
    - shape: Shape object whose boundary edges to plot
    - hex_size: circumradius of hexagons
    - color: color for edge lines (if None, uses shape's jittered color)
    - alpha: transparency for edges (0-1)
    - width: line width for boundary edges
    - jitter_amount: amount to jitter color if using shape's color
    """
    from javelance.shapes import Shape

    # Get boundary edges
    boundary_edges = shape.boundary_edges()

    if len(boundary_edges) == 0:
        return fig

    # Determine edge color
    if color is None:
        # Use shape's jittered color
        edge_color = shape.jittered_color(jitter_amount=jitter_amount)
    else:
        edge_color = color

    # Convert edge color to rgba format with alpha
    if edge_color.startswith("rgb"):
        # Already in rgb format, convert to rgba
        edge_rgba_color = edge_color.replace("rgb", "rgba").replace(")", f", {alpha})")
    else:
        # Named color, use directly with opacity parameter
        edge_rgba_color = edge_color

    # Plot each boundary edge
    for edge in boundary_edges:
        hex1 = edge[0]
        hex2 = edge[1]

        # Get the shared edge vertices
        v1, v2 = _shared_edge_vertices(hex1, hex2, hex_size)

        # Plot the edge without jitter (on the actual boundary)
        fig.add_trace(
            go.Scatter(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                mode="lines",
                line=dict(color=edge_rgba_color, width=width),
                opacity=alpha,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return fig


def plot_small_shape(fig, nodes, edges, hex_size=1.0, jitter=0.1, alpha=0.4):
    """
    Plot a small shape (4 nodes or fewer) on the hexagonal grid.
    Color is determined by the number of nodes:
    - 4 nodes: random shade of green
    - 3 nodes: blue
    - 2 nodes: red
    - 1 node: magenta

    Parameters:
    - fig: plotly Figure object to add traces to
    - nodes: Nx2 array of hex addresses (i, j), where N <= 4
    - edges: Mx2x2 array where each edge is [[i1, j1], [i2, j2]]
    - hex_size: circumradius of hexagons
    - jitter: amount to offset edges inward (as fraction of hex_size)
    - alpha: transparency for nodes and edges (0-1)
    """
    num_nodes = len(nodes)

    if num_nodes > 4:
        raise ValueError(
            f"plot_small_shape only supports shapes with 4 or fewer nodes, got {num_nodes}"
        )

    # Determine color based on number of nodes
    if num_nodes == 4:
        # Random shade of green
        r = np.random.randint(0, 100)
        g = np.random.randint(150, 256)
        b = np.random.randint(0, 100)
        color = f"rgb({r},{g},{b})"
    elif num_nodes == 3:
        # Random shade of blue
        r = np.random.randint(0, 100)
        g = np.random.randint(0, 100)
        b = np.random.randint(150, 256)
        color = f"rgb({r},{g},{b})"
    elif num_nodes == 2:
        # Random shade of red
        r = np.random.randint(150, 256)
        g = np.random.randint(0, 100)
        b = np.random.randint(0, 100)
        color = f"rgb({r},{g},{b})"
    elif num_nodes == 1:
        # Random shade of magenta
        r = np.random.randint(150, 256)
        g = np.random.randint(0, 100)
        b = np.random.randint(150, 256)
        color = f"rgb({r},{g},{b})"
    else:
        # 0 nodes - shouldn't happen but handle it
        color = "gray"

    return plot_shape(
        fig,
        nodes,
        edges,
        hex_size=hex_size,
        node_color=color,
        edge_color=color,
        jitter=jitter,
        alpha=alpha,
    )


def plot_javelance(regions=None) -> Figure:
    if regions is None:
        regions = list(JAVELANCE_REGIONS.values())

    # Create a grid large enough for the Javelance
    fig = plot_shape_hexes(JAVELANCE_GRID_SHAPE, label_hexes=False)

    # Plot JAVELANCE_FORBIDDEN
    plot_shape(
        fig,
        JAVELANCE_FORBIDDEN.nodes,
        JAVELANCE_FORBIDDEN.edges,
        node_color=JAVELANCE_FORBIDDEN.mean_color,
        edge_color=JAVELANCE_FORBIDDEN.mean_color,
    )

    # Plot JAVELANCE
    for r in regions:
        r.plot(fig, plot_boundary=True)

    JAVELANCE.difference(union_shapes(regions)).plot(fig)
    return fig


def plot_packing_solution(
    targeted_regions: list[Shape], solution: PackingSolution, title: str = None
) -> Figure:
    # Create visualization
    fig = plot_javelance(targeted_regions)

    if title:
        # Count the number of lines in the title (br tags + 1)
        num_lines = title.count("<br>") + 1
        # Increase top margin based on number of title lines
        # Base margin of 100 + 30 pixels per additional line
        top_margin = 100 + (num_lines - 1) * 30

        fig.update_layout(title=title, margin=dict(t=top_margin))

    # Plot each placed piece with a distinct jittered color
    for i, (name, shape) in enumerate(solution.placements):
        # Use different base colors for different piece types
        if name == "DOODAD":
            base_color = "rgb(0, 200, 0)"  # Green
        elif name == "GIZMO":
            base_color = "rgb(0, 0, 200)"  # Blue
        else:  # SPROCKET
            base_color = "rgb(200, 0, 200)"  # Magenta

        # Create a shape with the base color to use jittered_color
        colored_shape = Shape(
            nodes=shape.nodes, edges=shape.edges, mean_color=base_color
        )
        color = colored_shape.jittered_color(jitter_amount=30)

        plot_shape(
            fig,
            shape.nodes,
            shape.edges,
            node_color=color,
            edge_color=color,
            alpha=0.8,
            labels=lambda i_, j_: f"{i}",
        )
    return fig
