import numpy as np
import pytest

from javelance.shapes import Shape, DOODADS, GIZMOS, SPROCKETS
from javelance.plotting import plot_hex_grid, plot_shape


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
    edges = np.array(
        [
            [[0, 0], [1, 0]],
            [[1, 0], [0, 0]],  # Same edge as above, just reversed
        ]
    )

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


def test_shape_has_mean_color():
    """Test that Shape has a mean_color field with a default value."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    assert hasattr(shape, "mean_color")
    assert shape.mean_color == "rgb(128, 128, 128)"


def test_shape_custom_mean_color():
    """Test that Shape accepts a custom mean_color."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(100, 150, 200)")

    assert shape.mean_color == "rgb(100, 150, 200)"


def test_jittered_color_format():
    """Test that jittered_color returns a valid rgb() string."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(128, 128, 128)")

    jittered = shape.jittered_color()

    # Check format
    assert jittered.startswith("rgb(")
    assert jittered.endswith(")")

    # Extract and validate RGB values
    rgb_str = jittered[4:-1]
    r, g, b = map(int, rgb_str.split(","))
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


def test_jittered_color_clamping():
    """Test that jittered_color clamps values to valid range."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])

    # Test clamping at lower bound
    shape_low = Shape(nodes=nodes, edges=edges, mean_color="rgb(10, 10, 10)")
    for _ in range(10):
        jittered = shape_low.jittered_color(jitter_amount=50)
        rgb_str = jittered[4:-1]
        r, g, b = map(int, rgb_str.split(","))
        assert r >= 0 and g >= 0 and b >= 0

    # Test clamping at upper bound
    shape_high = Shape(nodes=nodes, edges=edges, mean_color="rgb(245, 245, 245)")
    for _ in range(10):
        jittered = shape_high.jittered_color(jitter_amount=50)
        rgb_str = jittered[4:-1]
        r, g, b = map(int, rgb_str.split(","))
        assert r <= 255 and g <= 255 and b <= 255


def test_jittered_color_with_rgba():
    """Test that jittered_color works with rgba format."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgba(128, 128, 128, 0.5)")

    jittered = shape.jittered_color()

    # Should return rgb format even if input is rgba
    assert jittered.startswith("rgb(")
    assert jittered.endswith(")")


def test_named_color_converted_to_rgb():
    """Test that named colors are converted to rgb format."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="blue")

    # After initialization, mean_color should be converted to rgb format
    assert shape.mean_color.startswith("rgb(")
    assert shape.mean_color.endswith(")")

    # Should be able to jitter it now
    jittered = shape.jittered_color()
    assert jittered.startswith("rgb(")


def test_invalid_color_name():
    """Test that invalid color names raise an error."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])

    with pytest.raises(ValueError, match="mean_color must be in 'rgb"):
        Shape(nodes=nodes, edges=edges, mean_color="not_a_real_color_name")


def test_rotations_preserve_mean_color():
    """Test that rotations preserve the mean_color."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(100, 150, 200)")

    rotations = shape.rotations()

    for rotated in rotations:
        assert rotated.mean_color == "rgb(100, 150, 200)"


def test_bounding_addresses():
    """Test that bounding_addresses returns correct min and max coordinates."""
    nodes = np.array([[1, 2], [5, 3], [2, 7]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    min_coords, max_coords = shape.bounding_addresses()

    assert np.array_equal(min_coords, np.array([1, 2]))
    assert np.array_equal(max_coords, np.array([5, 7]))


def test_bounding_addresses_single_node():
    """Test bounding_addresses with a single node."""
    nodes = np.array([[3, 4]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    min_coords, max_coords = shape.bounding_addresses()

    assert np.array_equal(min_coords, np.array([3, 4]))
    assert np.array_equal(max_coords, np.array([3, 4]))


def test_translate():
    """Test that translate shifts the shape correctly."""
    nodes = np.array([[1, 2], [2, 2]])  # Adjacent hexagons
    edges = np.array([[[1, 2], [2, 2]]])  # Valid edge (offset (1, 0))
    shape = Shape(nodes=nodes, edges=edges, mean_color="blue")

    displacement = np.array([10, 20])
    translated = shape.translate(displacement)

    expected_nodes = np.array([[11, 22], [12, 22]])
    expected_edges = np.array([[[11, 22], [12, 22]]])

    assert np.array_equal(translated.nodes, expected_nodes)
    assert np.array_equal(translated.edges, expected_edges)
    assert translated.mean_color == shape.mean_color


def test_translate_preserves_color():
    """Test that translate preserves mean_color."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="red")

    translated = shape.translate(np.array([5, 5]))

    assert translated.mean_color == shape.mean_color


def test_translate_inverse():
    """Test that translate(translate(x), -x) equals the original shape."""
    nodes = np.array([[1, 2], [2, 2], [3, 2]])  # Line of adjacent hexagons
    edges = np.array([[[1, 2], [2, 2]], [[2, 2], [3, 2]]])  # Valid edges
    shape = Shape(nodes=nodes, edges=edges, mean_color="green")

    displacement = np.array([7, -3])

    # Translate forward then backward
    translated_forward = shape.translate(displacement)
    translated_back = translated_forward.translate(-displacement)

    # Should equal original
    assert np.array_equal(translated_back.nodes, shape.nodes)
    assert np.array_equal(translated_back.edges, shape.edges)
    assert translated_back.mean_color == shape.mean_color


def test_translate_with_negative_displacement():
    """Test translate with negative displacement."""
    nodes = np.array([[5, 5], [6, 5]])  # Adjacent hexagons
    edges = np.array([[[5, 5], [6, 5]]])  # Valid edge (offset (1, 0))
    shape = Shape(nodes=nodes, edges=edges)

    translated = shape.translate(np.array([-2, -3]))

    expected_nodes = np.array([[3, 2], [4, 2]])
    assert np.array_equal(translated.nodes, expected_nodes)


def test_originated():
    """Test that originated shifts shape to have minimum coords at origin."""
    nodes = np.array([[3, 5], [4, 5], [4, 6]])  # Adjacent hexagons
    edges = np.array([[[3, 5], [4, 5]], [[4, 5], [4, 6]]])  # Valid edges
    shape = Shape(nodes=nodes, edges=edges)

    originated = shape.originated()

    # Minimum should be [0, 0]
    min_coords, _ = originated.bounding_addresses()
    assert np.array_equal(min_coords, np.array([0, 0]))

    # Shape should be shifted by -[3, 5]
    expected_nodes = np.array([[0, 0], [1, 0], [1, 1]])
    assert np.array_equal(originated.nodes, expected_nodes)


def test_originated_already_at_origin():
    """Test originated when shape is already at origin."""
    nodes = np.array([[0, 0], [1, 0], [1, 1]])  # Adjacent hexagons
    edges = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]]])  # Valid edges
    shape = Shape(nodes=nodes, edges=edges)

    originated = shape.originated()

    # Should be unchanged
    assert np.array_equal(originated.nodes, shape.nodes)
    assert np.array_equal(originated.edges, shape.edges)


def test_originated_with_negative_coords():
    """Test originated with negative coordinates."""
    nodes = np.array([[-5, -3], [-4, -3], [-4, -2]])  # Adjacent hexagons
    edges = np.array([[[-5, -3], [-4, -3]], [[-4, -3], [-4, -2]]])  # Valid edges
    shape = Shape(nodes=nodes, edges=edges)

    originated = shape.originated()

    # Minimum should be [0, 0]
    min_coords, _ = originated.bounding_addresses()
    assert np.array_equal(min_coords, np.array([0, 0]))

    # Shape should be shifted by -[-5, -3] = [5, 3]
    expected_nodes = np.array([[0, 0], [1, 0], [1, 1]])
    assert np.array_equal(originated.nodes, expected_nodes)


def test_originated_rotations_count():
    """Test that originated_rotations returns 6 shapes."""
    nodes = np.array([[1, 2], [2, 2]])  # Adjacent hexagons
    edges = np.array([[[1, 2], [2, 2]]])  # Valid edge
    shape = Shape(nodes=nodes, edges=edges)

    originated_rots = shape.originated_rotations()

    assert len(originated_rots) == 6


def test_originated_rotations_all_at_origin():
    """Test that all originated rotations have minimum coords at origin."""
    nodes = np.array([[5, 3], [6, 3], [6, 4]])  # Adjacent hexagons
    edges = np.array([[[5, 3], [6, 3]], [[6, 3], [6, 4]]])  # Valid edges
    shape = Shape(nodes=nodes, edges=edges)

    originated_rots = shape.originated_rotations()

    for rotated in originated_rots:
        min_coords, _ = rotated.bounding_addresses()
        assert np.array_equal(min_coords, np.array([0, 0])), (
            f"Rotation not at origin: min_coords = {min_coords}"
        )


def test_originated_rotations_preserve_color():
    """Test that originated_rotations preserves mean_color."""
    nodes = np.array([[1, 2], [2, 2]])  # Adjacent hexagons
    edges = np.array([[[1, 2], [2, 2]]])  # Valid edge
    shape = Shape(nodes=nodes, edges=edges, mean_color="purple")

    originated_rots = shape.originated_rotations()

    for rotated in originated_rots:
        assert rotated.mean_color == shape.mean_color


def test_equivalent_same_shape():
    """Test that a shape is equivalent to itself."""
    nodes = np.array([[1, 2], [2, 2], [2, 3]])
    edges = np.array([[[1, 2], [2, 2]], [[2, 2], [2, 3]]])
    shape1 = Shape(nodes=nodes, edges=edges)
    shape2 = Shape(nodes=nodes, edges=edges)

    assert shape1.equivalent(shape2)


def test_equivalent_translated_shapes():
    """Test that translated versions of a shape are equivalent."""
    nodes = np.array([[0, 0], [1, 0], [1, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]]])
    shape1 = Shape(nodes=nodes, edges=edges)

    # Translate the shape
    shape2 = shape1.translate(np.array([5, 3]))

    # They should be equivalent (after origination)
    assert shape1.equivalent(shape2)


def test_equivalent_different_node_order():
    """Test that shapes with same nodes in different order are equivalent."""
    nodes1 = np.array([[0, 0], [1, 0], [1, 1]])
    nodes2 = np.array([[1, 1], [0, 0], [1, 0]])  # Same nodes, different order
    edges1 = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]]])
    edges2 = np.array(
        [[[1, 0], [1, 1]], [[0, 0], [1, 0]]]
    )  # Same edges, different order

    shape1 = Shape(nodes=nodes1, edges=edges1)
    shape2 = Shape(nodes=nodes2, edges=edges2)

    assert shape1.equivalent(shape2)


def test_not_equivalent_different_nodes():
    """Test that shapes with different nodes are not equivalent."""
    nodes1 = np.array([[0, 0], [1, 0]])
    nodes2 = np.array([[0, 0], [0, 1]])  # Different second node
    edges1 = np.array([[[0, 0], [1, 0]]])
    edges2 = np.array([[[0, 0], [0, 1]]])

    shape1 = Shape(nodes=nodes1, edges=edges1)
    shape2 = Shape(nodes=nodes2, edges=edges2)

    assert not shape1.equivalent(shape2)


def test_not_equivalent_different_edges():
    """Test that shapes with same nodes but different edges are not equivalent."""
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges1 = np.array([[[0, 0], [1, 0]]])
    edges2 = np.array([[[0, 0], [0, 1]]])

    shape1 = Shape(nodes=nodes, edges=edges1)
    shape2 = Shape(nodes=nodes, edges=edges2)

    assert not shape1.equivalent(shape2)


def test_unique_originated_rotations_no_symmetry():
    """Test unique_originated_rotations with asymmetric shape (should return 6)."""
    # Create an L-shaped pattern with no rotational symmetry
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges)

    unique_rots = shape.unique_originated_rotations()

    # Asymmetric shape should have 6 unique rotations
    assert len(unique_rots) == 6


def test_unique_originated_rotations_with_symmetry():
    """Test unique_originated_rotations with symmetric shape (should return fewer than 6)."""
    # Create a line of 2 hexagons (180° rotational symmetry)
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    unique_rots = shape.unique_originated_rotations()

    # Line has 180° symmetry, so should have only 3 unique rotations
    assert len(unique_rots) == 3


def test_unique_originated_rotations_all_unique():
    """Test that all returned rotations are actually unique."""
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges)

    unique_rots = shape.unique_originated_rotations()

    # Check that no two rotations are equivalent to each other
    for i in range(len(unique_rots)):
        for j in range(i + 1, len(unique_rots)):
            assert not unique_rots[i].equivalent(unique_rots[j]), (
                f"Rotations {i} and {j} are equivalent but both in unique list"
            )


def test_unique_originated_rotations_preserve_color():
    """Test that unique_originated_rotations preserves mean_color."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="cyan")

    unique_rots = shape.unique_originated_rotations()

    for rotated in unique_rots:
        assert rotated.mean_color == shape.mean_color


def test_plot_doodads_rotations():
    """Visual test: Plot all rotations of DOODADS shapes."""
    fig = plot_hex_grid(25, 15)

    for shape_idx, shape in enumerate(DOODADS):
        unique_rots = shape.unique_originated_rotations()

        # Arrange rotations in a 2x3 grid
        for rot_idx, rotated in enumerate(unique_rots):
            # Calculate offset for this rotation
            row = rot_idx // 3
            col = rot_idx % 3

            # Get bounding box to ensure proper spacing
            min_coords, max_coords = rotated.bounding_addresses()
            shape_width = max_coords[0] - min_coords[0] + 1
            shape_height = max_coords[1] - min_coords[1] + 1

            # Add 1 hex spacing between shapes
            base_offset = np.array([col * (shape_width + 1), row * (shape_height + 1)])

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(
                fig,
                positioned.nodes,
                positioned.edges,
                node_color=color,
                edge_color=color,
            )

    fig.show()


def test_plot_gizmos_rotations():
    """Visual test: Plot all rotations of GIZMOS shapes."""
    shapes_per_row = 2
    rots_per_row = 3  # 3 rotations per row within each shape

    # Calculate maximum bounding box size for all rotations of all shapes
    max_width = 0
    max_height = 0
    for shape in GIZMOS:
        for rotated in shape.unique_originated_rotations():
            min_c, max_c = rotated.bounding_addresses()
            width = max_c[0] - min_c[0] + 1
            height = max_c[1] - min_c[1] + 1
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    # Calculate grid size
    grid_width = shapes_per_row * rots_per_row * (max_width + 1) + 2
    grid_height = ((len(GIZMOS) + shapes_per_row - 1) // shapes_per_row) * 2 * (
        max_height + 1
    ) + 2

    fig = plot_hex_grid(grid_width, grid_height)

    for shape_idx, shape in enumerate(GIZMOS):
        unique_rots = shape.unique_originated_rotations()

        # Calculate base offset for this shape's block
        shape_row = shape_idx // shapes_per_row
        shape_col = shape_idx % shapes_per_row

        for rot_idx, rotated in enumerate(unique_rots):
            # Position within the shape's grid (2 rows, 3 cols)
            rot_row = rot_idx // rots_per_row
            rot_col = rot_idx % rots_per_row

            # Calculate global offset
            base_offset = np.array(
                [
                    shape_col * rots_per_row * (max_width + 1)
                    + rot_col * (max_width + 1),
                    shape_row * 2 * (max_height + 1) + rot_row * (max_height + 1),
                ]
            )

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(
                fig,
                positioned.nodes,
                positioned.edges,
                node_color=color,
                edge_color=color,
            )

    fig.show()


def test_plot_sprockets_rotations():
    """Visual test: Plot all rotations of SPROCKETS shapes."""
    shapes_per_row = 2
    rots_per_row = 3  # 3 rotations per row within each shape

    # Calculate maximum bounding box size for all rotations of all shapes
    max_width = 0
    max_height = 0
    for shape in SPROCKETS:
        for rotated in shape.unique_originated_rotations():
            min_c, max_c = rotated.bounding_addresses()
            width = max_c[0] - min_c[0] + 1
            height = max_c[1] - min_c[1] + 1
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    # Calculate grid size
    grid_width = shapes_per_row * rots_per_row * (max_width + 1) + 2
    grid_height = ((len(SPROCKETS) + shapes_per_row - 1) // shapes_per_row) * 2 * (
        max_height + 1
    ) + 2

    fig = plot_hex_grid(grid_width, grid_height)

    for shape_idx, shape in enumerate(SPROCKETS):
        unique_rots = shape.unique_originated_rotations()

        # Calculate base offset for this shape's block
        shape_row = shape_idx // shapes_per_row
        shape_col = shape_idx % shapes_per_row

        for rot_idx, rotated in enumerate(unique_rots):
            # Position within the shape's grid (2 rows, 3 cols)
            rot_row = rot_idx // rots_per_row
            rot_col = rot_idx % rots_per_row

            # Calculate global offset
            base_offset = np.array(
                [
                    shape_col * rots_per_row * (max_width + 1)
                    + rot_col * (max_width + 1),
                    shape_row * 2 * (max_height + 1) + rot_row * (max_height + 1),
                ]
            )

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(
                fig,
                positioned.nodes,
                positioned.edges,
                node_color=color,
                edge_color=color,
            )

    fig.show()


def test_box():
    """Test that box creates a rectangular box of nodes."""
    shape = Shape.box(width=3, height=2, mean_color="rgb(100, 100, 100)")

    # Should have 3*2 = 6 nodes
    assert len(shape.nodes) == 6

    # Nodes should be at positions (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
    expected_nodes = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}
    assert set(map(tuple, shape.nodes)) == expected_nodes

    # Should have no edges
    assert len(shape.edges) == 0

    # Should have correct color
    assert shape.mean_color == "rgb(100, 100, 100)"


def test_vertical_box():
    """Test that vertical_box creates a vertically oriented box."""
    shape = Shape.vertical_box(width=2, height=3, mean_color="rgb(150, 150, 150)")

    # Should have 2*3 = 6 nodes
    assert len(shape.nodes) == 6

    # For vertical_box, rows are offset by (-i//2, i)
    # Row 0 (i=0): offset (-0//2, 0) = (0, 0) → nodes at (0,0), (1,0)
    # Row 1 (i=1): offset (-1//2, 1) = (0, 1) → nodes at (0,1), (1,1)
    # Row 2 (i=2): offset (-2//2, 2) = (-1, 2) → nodes at (-1,2), (0,2)
    expected_nodes = {(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2), (0, 2)}
    assert set(map(tuple, shape.nodes)) == expected_nodes

    # Should have no edges
    assert len(shape.edges) == 0


def test_vertical_box_offset_pattern():
    """Test that vertical_box has the correct horizontal offset pattern for vertical stacking."""
    # Create a taller vertical box to verify the offset pattern
    shape = Shape.vertical_box(width=3, height=4, mean_color="rgb(150, 150, 150)")

    # Should have 3*4 = 12 nodes
    assert len(shape.nodes) == 12

    # For vertical_box with width=3, height=4:
    # Row 0 (i=0): offset (-0//2, 0) = (0, 0) → nodes at (0,0), (1,0), (2,0)
    # Row 1 (i=1): offset (-1//2, 1) = (0, 1) → nodes at (0,1), (1,1), (2,1)
    # Row 2 (i=2): offset (-2//2, 2) = (-1, 2) → nodes at (-1,2), (0,2), (1,2)
    # Row 3 (i=3): offset (-3//2, 3) = (-1, 3) → nodes at (-1,3), (0,3), (1,3)
    expected_nodes = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (-1, 2),
        (0, 2),
        (1, 2),
        (-1, 3),
        (0, 3),
        (1, 3),
    }
    assert set(map(tuple, shape.nodes)) == expected_nodes


def test_node_set():
    """Test that node_set returns nodes as a set of tuples."""
    nodes = np.array([[1, 2], [3, 4], [5, 6]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    node_set = shape.node_set()

    assert isinstance(node_set, set)
    assert node_set == {(1, 2), (3, 4), (5, 6)}


def test_edge_set():
    """Test that edge_set returns normalized edges as a set."""
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    edges = np.array([[[0, 0], [1, 0]], [[2, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    edge_set = shape.edge_set()

    assert isinstance(edge_set, set)
    # Edges should be normalized (nodes sorted)
    assert edge_set == {((0, 0), (1, 0)), ((1, 0), (2, 0))}


def test_edge_set_normalization():
    """Test that edge_set normalizes edges by sorting nodes."""
    nodes = np.array([[0, 0], [1, 0]])
    # Create two edges that are the same but in different order
    edges = np.array([[[1, 0], [0, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    edge_set = shape.edge_set()

    # Should be normalized to ((0, 0), (1, 0))
    assert edge_set == {((0, 0), (1, 0))}


def test_from_sets():
    """Test that from_sets constructs a Shape from sets of nodes and edges."""
    nodes_set = {(0, 0), (1, 0), (2, 0)}
    edges_set = {((0, 0), (1, 0)), ((1, 0), (2, 0))}
    shape = Shape.from_sets(
        nodes=nodes_set, edges=edges_set, mean_color="rgb(200, 200, 200)"
    )

    # Should have correct nodes
    assert set(map(tuple, shape.nodes)) == nodes_set

    # Should have correct edges (normalized)
    assert shape.edge_set() == edges_set

    # Should have correct color
    assert shape.mean_color == "rgb(200, 200, 200)"


def test_from_sets_empty():
    """Test that from_sets handles empty sets."""
    shape = Shape.from_sets(nodes=set(), edges=set())

    assert len(shape.nodes) == 0
    assert len(shape.edges) == 0


def test_from_sets_preserves_default_color():
    """Test that from_sets uses default color when mean_color is None."""
    nodes_set = {(0, 0)}
    shape = Shape.from_sets(nodes=nodes_set, edges=set(), mean_color=None)

    assert shape.mean_color == "rgb(128, 128, 128)"


def test_bounding_box_method():
    """Test that bounding_box returns a Shape covering the bounding addresses."""
    nodes = np.array([[1, 2], [3, 4], [2, 3]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(100, 150, 200)")

    bbox = shape.bounding_box()

    # Bounding box should cover from (1, 2) to (3, 4)
    # That's a 3x3 box: width=3, height=3
    expected_nodes = set()
    for i in range(1, 4):
        for j in range(2, 5):
            expected_nodes.add((i, j))

    assert set(map(tuple, bbox.nodes)) == expected_nodes
    assert len(bbox.edges) == 0  # box() creates no edges
    assert bbox.mean_color == "rgb(100, 150, 200)"


def test_difference():
    """Test that difference returns the set difference of two shapes."""
    # Create two shapes with overlapping nodes
    nodes1 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    nodes2 = np.array([[1, 0], [2, 0]])
    edges1 = np.array([[[0, 0], [1, 0]], [[1, 0], [2, 0]], [[2, 0], [3, 0]]])
    edges2 = np.array([[[1, 0], [2, 0]]])

    shape1 = Shape(nodes=nodes1, edges=edges1, mean_color="rgb(255, 0, 0)")
    shape2 = Shape(nodes=nodes2, edges=edges2)

    diff = shape1.difference(shape2)

    # Should have nodes [0,0] and [3,0] (nodes in shape1 but not shape2)
    assert diff.node_set() == {(0, 0), (3, 0)}

    # Should have edges [[0,0],[1,0]] and [[2,0],[3,0]] (edges in shape1 but not shape2)
    expected_edges = {((0, 0), (1, 0)), ((2, 0), (3, 0))}
    assert diff.edge_set() == expected_edges

    # Should preserve color from first shape
    assert diff.mean_color == "rgb(255, 0, 0)"


def test_difference_no_overlap():
    """Test difference when shapes don't overlap."""
    nodes1 = np.array([[0, 0], [1, 0]])
    nodes2 = np.array([[5, 5], [6, 5]])
    edges1 = np.array([[[0, 0], [1, 0]]])
    edges2 = np.array([[[5, 5], [6, 5]]])

    shape1 = Shape(nodes=nodes1, edges=edges1)
    shape2 = Shape(nodes=nodes2, edges=edges2)

    diff = shape1.difference(shape2)

    # Should have all nodes from shape1
    assert diff.node_set() == {(0, 0), (1, 0)}
    assert diff.edge_set() == {((0, 0), (1, 0))}


def test_negative_nodes():
    """Test that negative_nodes returns nodes in bounding box but not in shape."""
    # Create an L-shape
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    edges = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(50, 100, 150)")

    negative = shape.negative_nodes()

    # Bounding box is 2x2 (from (0,0) to (1,1))
    # Bounding box nodes: (0,0), (1,0), (0,1), (1,1)
    # Shape nodes: (0,0), (1,0), (0,1)
    # Negative nodes: (1,1)
    assert negative.node_set() == {(1, 1)}

    # Should have no edges (negative_nodes creates a shape with no edges)
    assert len(negative.edges) == 0

    # Should preserve color
    assert negative.mean_color == "rgb(50, 100, 150)"


def test_negative_nodes_full_box():
    """Test negative_nodes when shape fills its bounding box."""
    # Create a 2x2 box
    shape = Shape.box(width=2, height=2, mean_color="rgb(100, 100, 100)")

    negative = shape.negative_nodes()

    # Should have no nodes (all nodes in bounding box are in the shape)
    assert len(negative.nodes) == 0


def test_roundtrip_sets_conversion():
    """Test that converting to sets and back preserves the shape."""
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    edges = np.array([[[0, 0], [1, 0]], [[1, 0], [2, 0]]])
    original = Shape(nodes=nodes, edges=edges, mean_color="rgb(123, 45, 67)")

    # Convert to sets
    node_set = original.node_set()
    edge_set = original.edge_set()

    # Convert back
    reconstructed = Shape.from_sets(
        nodes=node_set, edges=edge_set, mean_color=original.mean_color
    )

    # Should be equivalent
    assert reconstructed.node_set() == original.node_set()
    assert reconstructed.edge_set() == original.edge_set()
    assert reconstructed.mean_color == original.mean_color


def test_adjacent_nodes_single_node():
    """Test that adjacent_nodes returns all 6 adjacent hexes for a single node."""
    nodes = np.array([[0, 0]])
    adjacent = Shape.adjacent_nodes(nodes)

    # Should have 6 adjacent nodes
    assert len(adjacent) == 6

    # Should include all valid adjacency offsets from (0,0)
    expected = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}
    assert set(map(tuple, adjacent)) == expected


def test_adjacent_nodes_multiple_nodes():
    """Test that adjacent_nodes returns all adjacent hexes for multiple nodes."""
    nodes = np.array([[0, 0], [1, 0]])
    adjacent = Shape.adjacent_nodes(nodes)

    # Should have 12 nodes (6 for each input node)
    assert len(adjacent) == 12

    # For (0,0): (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # For (1,0): (2,0), (0,0), (1,1), (1,-1), (2,-1), (0,1)
    expected = {
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (2, 0),
        (0, 0),
        (1, 1),
        (0, 1),
        (2, -1),
    }
    assert set(map(tuple, adjacent)) == expected


def test_adjacent_nodes_empty():
    """Test that adjacent_nodes handles empty input."""
    nodes = np.empty((0, 2), dtype=int)
    adjacent = Shape.adjacent_nodes(nodes)

    assert len(adjacent) == 0
    assert adjacent.shape == (0, 2)


def test_adjacent():
    """Test that adjacent() returns a Shape with all adjacent nodes."""
    nodes = np.array([[0, 0]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges, mean_color="rgb(100, 100, 100)")

    adjacent_shape = shape.adjacent()

    # Should have 6 adjacent nodes
    assert len(adjacent_shape.nodes) == 6

    # Should have the correct adjacent nodes
    expected_nodes = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}
    assert set(map(tuple, adjacent_shape.nodes)) == expected_nodes

    # Should have no edges
    assert len(adjacent_shape.edges) == 0

    # Should preserve color
    assert adjacent_shape.mean_color == "rgb(100, 100, 100)"


def test_adjacent_multiple_nodes():
    """Test adjacent() with multiple nodes removes duplicates."""
    # Two adjacent hexes
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    adjacent_shape = shape.adjacent()

    # Adjacent nodes should include neighbors of both, with duplicates removed
    # From (0,0): (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # From (1,0): (2,0), (0,0), (1,1), (1,-1), (2,-1), (0,1)
    # Combined unique: (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1), (2,-1), (2,0)
    expected = {
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, -1),
        (2, 0),
    }
    assert set(map(tuple, adjacent_shape.nodes)) == expected


def test_as_node_set():
    """Test that as_node_set converts nodes array to set."""
    nodes = np.array([[1, 2], [3, 4], [5, 6]])
    node_set = Shape.as_node_set(nodes)

    assert isinstance(node_set, set)
    assert node_set == {(1, 2), (3, 4), (5, 6)}


def test_as_node_set_empty():
    """Test that as_node_set handles empty arrays."""
    nodes = np.empty((0, 2), dtype=int)
    node_set = Shape.as_node_set(nodes)

    assert node_set == set()


def test_as_edge_set():
    """Test that as_edge_set converts edges array to normalized set."""
    edges = np.array([[[0, 0], [1, 0]], [[2, 0], [1, 0]]])
    edge_set = Shape.as_edge_set(edges)

    assert isinstance(edge_set, set)
    # Edges should be normalized
    assert edge_set == {((0, 0), (1, 0)), ((1, 0), (2, 0))}


def test_as_edge_set_normalizes():
    """Test that as_edge_set normalizes edges."""
    edges = np.array([[[1, 0], [0, 0]]])
    edge_set = Shape.as_edge_set(edges)

    # Should be normalized to ((0,0), (1,0))
    assert edge_set == {((0, 0), (1, 0))}


def test_as_edge_set_empty():
    """Test that as_edge_set handles empty arrays."""
    edges = np.empty((0, 2, 2), dtype=int)
    edge_set = Shape.as_edge_set(edges)

    assert edge_set == set()


def test_interior_edges_simple():
    """Test that edges_full returns all valid edges for a simple shape."""
    # Three nodes in a line: (0,0) - (1,0) - (2,0)
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    full_edges = shape.interior_edges()

    # Should have edges between adjacent hexes
    expected = {((0, 0), (1, 0)), ((1, 0), (2, 0))}
    assert full_edges == expected


def test_interior_edges_square():
    """Test edges_full on a 2x2 box."""
    # Create a 2x2 box
    shape = Shape.box(width=2, height=2)

    full_edges = shape.interior_edges()

    # A 2x2 box has nodes: (0,0), (1,0), (0,1), (1,1)
    # Valid adjacencies in pointy-top hex grid:
    # (0,0) - (1,0): offset (1,0)
    # (0,0) - (0,1): offset (0,1)
    # (1,0) - (1,1): offset (0,1)
    # (1,0) - (0,1): offset (-1,1)
    # (0,1) - (1,1): offset (1,0)
    expected = {
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
    }
    assert full_edges == expected


def test_interior_edges_isolated_nodes():
    """Test edges_full with isolated nodes."""
    # Two nodes that are not adjacent
    nodes = np.array([[0, 0], [5, 5]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    full_edges = shape.interior_edges()

    # No edges since nodes are not adjacent
    assert full_edges == set()


def test_interior_edges_single_node():
    """Test edges_full with a single node."""
    nodes = np.array([[0, 0]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    full_edges = shape.interior_edges()

    # No edges for a single node
    assert full_edges == set()


def test_plot_javelance():
    """Visual test: Plot JAVELANCE_PROTO and JAVELANCE shapes."""
    from javelance.shapes import JAVELANCE_FORBIDDEN, JAVELANCE

    # Create a grid large enough for both shapes
    fig = plot_hex_grid(50, 25)

    # Plot JAVELANCE_PROTO on the left
    plot_shape(
        fig,
        JAVELANCE_FORBIDDEN.nodes,
        JAVELANCE_FORBIDDEN.edges,
        node_color=JAVELANCE_FORBIDDEN.mean_color,
        edge_color=JAVELANCE_FORBIDDEN.mean_color,
    )

    # Plot JAVELANCE on the right (offset by 25 in x direction)
    javelance_offset = JAVELANCE.translate(np.array([0, 0]))
    plot_shape(
        fig,
        javelance_offset.nodes,
        javelance_offset.edges,
        node_color=javelance_offset.mean_color,
        edge_color=javelance_offset.mean_color,
    )

    fig.show()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
