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


if __name__ == "__main__":
    test_plot_rotations()
