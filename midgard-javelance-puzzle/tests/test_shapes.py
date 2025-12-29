import numpy as np
import pytest

from javelance.shapes import Shape


def test_valid_shape():
    """Test that a valid shape can be created."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)
    assert len(shape.nodes) == 2
    assert len(shape.edges) == 1


def test_nodes_not_2d():
    """Test that nodes must be a 2D array."""
    nodes = np.array([0, 0])  # 1D array
    edges = np.array([[[0, 0], [1, 0]]])

    with pytest.raises(ValueError, match="nodes must be a 2D array"):
        Shape(nodes=nodes, edges=edges)


def test_nodes_wrong_dimension_size():
    """Test that nodes must have size 2 in dimension 1."""
    nodes = np.array([[0, 0, 0], [1, 0, 0]])  # 3 columns instead of 2
    edges = np.array([[[0, 0], [1, 0]]])

    with pytest.raises(ValueError, match="nodes must have size 2 in dimension 1"):
        Shape(nodes=nodes, edges=edges)


def test_nodes_not_integer_type():
    """Test that nodes must have integer type."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])  # float instead of int
    edges = np.array([[[0, 0], [1, 0]]])

    with pytest.raises(ValueError, match="nodes must have integer type"):
        Shape(nodes=nodes, edges=edges)


def test_edges_not_3d():
    """Test that edges must be a 3D array."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[0, 0], [1, 0]])  # 2D array instead of 3D

    with pytest.raises(ValueError, match="edges must be a 3D array"):
        Shape(nodes=nodes, edges=edges)


def test_edges_wrong_dimension_1_size():
    """Test that edges must have size 2 in dimension 1."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0], [2, 0]]])  # 3 elements in dim 1 instead of 2

    with pytest.raises(ValueError, match="edges must have size 2 in dimension 1"):
        Shape(nodes=nodes, edges=edges)


def test_edges_wrong_dimension_2_size():
    """Test that edges must have size 2 in dimension 2."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0, 0], [1, 0, 0]]])  # 3 elements in dim 2 instead of 2

    with pytest.raises(ValueError, match="edges must have size 2 in dimension 2"):
        Shape(nodes=nodes, edges=edges)


def test_edges_not_integer_type():
    """Test that edges must have integer type."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0.0, 0.0], [1.0, 0.0]]])  # float instead of int

    with pytest.raises(ValueError, match="edges must have integer type"):
        Shape(nodes=nodes, edges=edges)


def test_duplicate_nodes():
    """Test that duplicate nodes are not allowed."""
    nodes = np.array([[0, 0], [1, 0], [0, 0]])  # [0, 0] appears twice
    edges = np.array([[[0, 0], [1, 0]]])

    with pytest.raises(ValueError, match="Duplicate nodes found"):
        Shape(nodes=nodes, edges=edges)


def test_duplicate_edges():
    """Test that duplicate edges are not allowed."""
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    edges = np.array([
        [[0, 0], [1, 0]],
        [[1, 0], [0, 0]],  # Same edge as above, just reversed
    ])

    with pytest.raises(ValueError, match="Duplicate edge found"):
        Shape(nodes=nodes, edges=edges)


def test_non_adjacent_hexagons():
    """Test that edges must connect adjacent hexagons."""
    nodes = np.array([[0, 0], [2, 0]])  # These hexagons are not adjacent
    edges = np.array([[[0, 0], [2, 0]]])

    with pytest.raises(ValueError, match="are not adjacent"):
        Shape(nodes=nodes, edges=edges)


def test_all_valid_adjacencies():
    """Test that all 6 valid adjacency directions work."""
    # Valid offsets: (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)

    # Test each valid offset
    for offset in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        nodes = np.array([[0, 0], [offset[0], offset[1]]])
        edges = np.array([[[0, 0], [offset[0], offset[1]]]])
        shape = Shape(nodes=nodes, edges=edges)
        assert len(shape.edges) == 1


def test_invalid_diagonal_adjacency():
    """Test that diagonal adjacencies that aren't valid are rejected."""
    # (1, 1) is not a valid adjacency offset
    nodes = np.array([[0, 0], [1, 1]])
    edges = np.array([[[0, 0], [1, 1]]])

    with pytest.raises(ValueError, match="are not adjacent"):
        Shape(nodes=nodes, edges=edges)


def test_rotations_count():
    """Test that rotations returns exactly 6 shapes."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    rotations = shape.rotations()
    assert len(rotations) == 6


def test_rotations_first_is_identity():
    """Test that the first rotation is the original shape."""
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges)

    rotations = shape.rotations()

    # First rotation should be identical to original
    assert np.array_equal(rotations[0].nodes, shape.nodes)
    assert np.array_equal(rotations[0].edges, shape.edges)


def test_rotations_180_degree():
    """Test that 180° rotation inverts coordinates."""
    nodes = np.array([[1, 2], [2, 2]])  # Adjacent hexagons
    edges = np.array([[[1, 2], [2, 2]]])  # Valid edge (offset is (1, 0))
    shape = Shape(nodes=nodes, edges=edges)

    rotations = shape.rotations()

    # 180° rotation is at index 3
    rotated_180 = rotations[3]

    # 180° rotation: (i, j) -> (-i, -j)
    expected_nodes = np.array([[-1, -2], [-2, -2]])
    assert np.array_equal(rotated_180.nodes, expected_nodes)


def test_rotations_all_valid():
    """Test that all rotations produce valid shapes."""
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges)

    rotations = shape.rotations()

    # All rotations should be valid (no exceptions raised during creation)
    for i, rotated in enumerate(rotations):
        assert len(rotated.nodes) == len(nodes), f"Rotation {i} has wrong node count"
        assert len(rotated.edges) == len(edges), f"Rotation {i} has wrong edge count"


def test_rotations_60_degree():
    """Test specific 60° rotation transformation."""
    nodes = np.array([[1, 0]])
    edges = np.empty((0, 2, 2), dtype=int)  # No edges for simplicity
    shape = Shape(nodes=nodes, edges=edges)

    rotations = shape.rotations()

    # 60° rotation: (1, 0) -> (0, 1)
    rotated_60 = rotations[1]
    expected_nodes = np.array([[0, 1]])
    assert np.array_equal(rotated_60.nodes, expected_nodes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
