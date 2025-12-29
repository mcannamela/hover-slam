import numpy as np
from javelance.plotting import plot_hex_grid, plot_shape


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


if __name__ == "__main__":
    test_plot_shape()
