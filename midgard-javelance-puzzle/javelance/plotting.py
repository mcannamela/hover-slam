import numpy as np
import plotly.graph_objects as go


def plot_hex_grid(I, J, hex_size=1.0, exclude=None):
    """
    Plot a hexagonal grid with pointy-top orientation.

    Parameters:
    - I: number of hexagons in the i direction (aligned with x-axis)
    - J: number of hexagons in the j direction (60° counterclockwise from x-axis)
    - hex_size: circumradius of each hexagon (distance from center to vertex)
    - exclude: nx2 array of (i, j) pairs to exclude from plotting (optional)

    The hexagons are indexed by (i, j) where:
    - i direction is aligned with the x-axis
    - j direction is 60° counterclockwise from the x-axis
    """
    # Convert exclude list to a set of tuples for fast lookup
    if exclude is not None:
        exclude_set = set(map(tuple, exclude))
    else:
        exclude_set = set()
    # For pointy-top hexagons with circumradius R:
    # - i offset vector: (sqrt(3) * R, 0)
    # - j offset vector: (sqrt(3)/2 * R, 3/2 * R)
    R = hex_size
    i_offset = np.array([np.sqrt(3) * R, 0])
    j_offset = np.array([np.sqrt(3)/2 * R, 3/2 * R])

    # Vertices of a pointy-top hexagon are at angles: 30°, 90°, 150°, 210°, 270°, 330°
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180

    fig = go.Figure()

    # Generate all hexagons
    for i in range(I):
        for j in range(J):
            # Calculate center position of hexagon (i, j)
            center = i * i_offset + j * j_offset
            cx, cy = center

            # Calculate vertices
            vertices_x = cx + R * np.cos(angles)
            vertices_y = cy + R * np.sin(angles)

            # Close the hexagon by adding the first vertex at the end
            vertices_x = np.append(vertices_x, vertices_x[0])
            vertices_y = np.append(vertices_y, vertices_y[0])

            # Check if this hexagon should be excluded (shaded grey)
            if (i, j) in exclude_set:
                fig.add_trace(go.Scatter(
                    x=vertices_x,
                    y=vertices_y,
                    mode='lines',
                    line=dict(color='black', width=1),
                    fill='toself',
                    fillcolor='lightgrey',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                # Add hexagon outline only
                fig.add_trace(go.Scatter(
                    x=vertices_x,
                    y=vertices_y,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Set equal aspect ratio and clean layout
    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def _hex_center(i, j, hex_size=1.0):
    """Calculate the center position of a hexagon at address (i, j)."""
    R = hex_size
    i_offset = np.array([np.sqrt(3) * R, 0])
    j_offset = np.array([np.sqrt(3)/2 * R, 3/2 * R])
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
        (1, 0): (330, 30),      # right edge
        (-1, 0): (150, 210),    # left edge
        (0, 1): (30, 90),       # upper-right edge
        (0, -1): (210, 270),    # lower-left edge
        (1, -1): (270, 330),    # lower-right edge
        (-1, 1): (90, 150),     # upper-left edge
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


def plot_shape(fig, nodes, edges, hex_size=1.0, node_color='red', edge_color='blue', jitter=0.1):
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
    """
    # Plot nodes
    if len(nodes) > 0:
        node_positions = np.array([_hex_center(i, j, hex_size) for i, j in nodes])
        fig.add_trace(go.Scatter(
            x=node_positions[:, 0],
            y=node_positions[:, 1],
            mode='markers',
            marker=dict(size=10, color=node_color, symbol='circle'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Plot edges
    for edge in edges:
        hex1 = tuple(edge[0])
        hex2 = tuple(edge[1])

        # Get the shared edge vertices
        v1, v2 = _shared_edge_vertices(hex1, hex2, hex_size)

        # Calculate center of the edge
        edge_center = (v1 + v2) / 2

        # Calculate inward jitter direction (toward the midpoint between hex centers)
        center1 = _hex_center(hex1[0], hex1[1], hex_size)
        center2 = _hex_center(hex2[0], hex2[1], hex_size)
        midpoint = (center1 + center2) / 2

        # Jitter toward the interior (perpendicular to the edge, toward midpoint)
        jitter_vec = midpoint - edge_center
        jitter_distance = np.linalg.norm(jitter_vec)
        if jitter_distance > 0:
            jitter_vec = jitter_vec / jitter_distance * jitter * hex_size

        # Apply jitter to edge vertices
        v1_jittered = v1 + jitter_vec
        v2_jittered = v2 + jitter_vec

        # Plot the edge
        fig.add_trace(go.Scatter(
            x=[v1_jittered[0], v2_jittered[0]],
            y=[v1_jittered[1], v2_jittered[1]],
            mode='lines',
            line=dict(color=edge_color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

    return fig
