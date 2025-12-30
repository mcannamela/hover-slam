import numpy as np
from javelance.plotting import plot_hex_grid, plot_shape, plot_small_shape


def test_plot_hex_grid_with_exclusions():
    """Test plotting a hexagonal grid with some hexagons excluded."""
    exclude_hexes = np.array([
        [1, 1],
        [2, 2],
        [3, 1],
    ])
    fig = plot_hex_grid(5, 4, exclude=exclude_hexes)
    fig.show()


def test_plot_shape():
    """Test plotting a shape on top of a hexagonal grid."""
    # Create the grid
    fig = plot_hex_grid(5, 4)

    # Define a shape with nodes and edges
    nodes = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
    ])

    edges = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [2, 2]],
        [[2, 2], [1, 2]],
        [[1, 2], [1, 1]],
    ])

    # Add the shape to the figure
    plot_shape(fig, nodes, edges)
    fig.show()


def test_plot_small_shapes():
    """Test plotting small shapes with different node counts."""
    # Create the grid
    fig = plot_hex_grid(6, 5)

    # 4-node shape (green)
    nodes_4 = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    edges_4 = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [2, 2]],
        [[2, 2], [1, 2]],
        [[1, 2], [1, 1]],
    ])
    plot_small_shape(fig, nodes_4, edges_4)

    # 3-node shape (blue)
    nodes_3 = np.array([[3, 1], [4, 1], [3, 2]])
    edges_3 = np.array([
        [[3, 1], [4, 1]],
        [[4, 1], [3, 2]],
        [[3, 2], [3, 1]],
    ])
    plot_small_shape(fig, nodes_3, edges_3)

    # 2-node shape (red)
    nodes_2 = np.array([[0, 3], [1, 3]])
    edges_2 = np.array([
        [[0, 3], [1, 3]],
    ])
    plot_small_shape(fig, nodes_2, edges_2)

    # 1-node shape (magenta)
    nodes_1 = np.array([[4, 3]])
    edges_1 = np.array([])  # No edges for a single node
    plot_small_shape(fig, nodes_1, edges_1)

    fig.show()


def test_plot_rotations():
    """Test plotting all rotations of a shape."""
    from javelance.shapes import Shape

    # Create a simple L-shaped pattern
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges)

    # Get all rotations
    rotations = shape.rotations()

    # Create a grid and plot all rotations
    fig = plot_hex_grid(8, 8)

    # Plot each rotation at a different location
    offsets = [
        (0, 0), (3, 0), (6, 0),
        (0, 3), (3, 3), (6, 3),
    ]

    for i, (rotated, offset) in enumerate(zip(rotations, offsets)):
        # Shift the nodes by the offset
        shifted_nodes = rotated.nodes + np.array(offset)
        shifted_edges = rotated.edges + np.array(offset)

        plot_small_shape(fig, shifted_nodes, shifted_edges)

    fig.show()


def test_plot_jittered_colors():
    """Test plotting shapes with jittered colors."""
    from javelance.shapes import Shape

    # Create a shape with a specific mean color
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(100, 150, 200)")

    # Create a grid and plot multiple instances with jittered colors
    fig = plot_hex_grid(8, 5)

    # Plot the same shape at different locations with jittered colors
    offsets = [
        (0, 0), (2, 0), (4, 0), (6, 0),
        (0, 2), (2, 2), (4, 2), (6, 2),
    ]

    for offset in offsets:
        # Shift the nodes by the offset
        shifted_nodes = shape.nodes + np.array(offset)
        shifted_edges = shape.edges + np.array(offset)

        # Get a jittered color
        color = shape.jittered_color(jitter_amount=30)

        # Plot with the jittered color
        plot_shape(fig, shifted_nodes, shifted_edges, node_color=color, edge_color=color)

    fig.show()


def test_plot_named_colors():
    """Test plotting shapes with named colors."""
    from javelance.shapes import Shape

    # Create shapes with various named colors
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])

    named_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    # Create a grid
    fig = plot_hex_grid(8, 5)

    # Plot shapes with different named colors
    for idx, color_name in enumerate(named_colors):
        # Calculate offset position
        row = idx // 4
        col = idx % 4
        offset = (col * 2, row * 2)

        # Create shape with named color
        shape = Shape(nodes=nodes, edges=edges, mean_color=color_name)

        # Shift the nodes by the offset
        shifted_nodes = shape.nodes + np.array(offset)
        shifted_edges = shape.edges + np.array(offset)

        # Plot with the shape's converted color
        plot_shape(fig, shifted_nodes, shifted_edges,
                  node_color=shape.mean_color, edge_color=shape.mean_color)

    fig.show()


def test_plot_shape_with_labels_dict():
    """Test plotting a shape with labels provided as a dict."""
    # Create the grid
    fig = plot_hex_grid(5, 4)

    # Define a shape with nodes and edges
    nodes = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
    ])

    edges = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [2, 2]],
        [[2, 2], [1, 2]],
        [[1, 2], [1, 1]],
    ])

    # Create labels as a dict mapping (i, j) to strings
    labels = {
        (1, 1): "A",
        (2, 1): "B",
        (2, 2): "C",
        (1, 2): "D",
    }

    # Add the shape with labels to the figure
    plot_shape(fig, nodes, edges, labels=labels)
    fig.show()


def test_plot_shape_with_labels_function():
    """Test plotting a shape with labels provided as a function."""
    # Create the grid
    fig = plot_hex_grid(5, 4)

    # Define a shape with nodes and edges
    nodes = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
    ])

    edges = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [2, 2]],
        [[2, 2], [1, 2]],
        [[1, 2], [1, 1]],
    ])

    # Create labels as a function that returns the coordinates
    def label_func(i, j):
        return f"{i},{j}"

    # Add the shape with labels to the figure
    plot_shape(fig, nodes, edges, labels=label_func)
    fig.show()


def test_plot_shape_with_labels_list():
    """Test plotting a shape with labels provided as a list."""
    # Create the grid
    fig = plot_hex_grid(5, 4)

    # Define a shape with nodes and edges
    nodes = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
    ])

    edges = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [2, 2]],
        [[2, 2], [1, 2]],
        [[1, 2], [1, 1]],
    ])

    # Create labels as a list parallel to nodes
    labels = ["1", "2", "3", "4"]

    # Add the shape with labels to the figure
    plot_shape(fig, nodes, edges, labels=labels)
    fig.show()


def test_plot_shape_with_numeric_labels():
    """Test plotting a shape with numeric labels."""
    # Create the grid
    fig = plot_hex_grid(6, 5)

    # Define a shape with nodes and edges
    nodes = np.array([
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 2],
        [2, 2],
        [3, 2],
    ])

    edges = np.array([
        [[1, 1], [2, 1]],
        [[2, 1], [3, 1]],
        [[1, 2], [2, 2]],
        [[2, 2], [3, 2]],
    ])

    # Create numeric labels (e.g., costs or weights)
    labels = [10, 20, 30, 15, 25, 35]

    # Add the shape with labels to the figure
    plot_shape(fig, nodes, edges, labels=labels, node_color="lightblue")
    fig.show()


if __name__ == "__main__":
    test_plot_shape_with_labels_dict()
