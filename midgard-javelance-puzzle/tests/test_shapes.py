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


def test_shape_has_mean_color():
    """Test that Shape has a mean_color field with a default value."""
    nodes = np.array([[0, 0], [1, 0]])
    edges = np.array([[[0, 0], [1, 0]]])
    shape = Shape(nodes=nodes, edges=edges)

    assert hasattr(shape, 'mean_color')
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


def test_bounding_box():
    """Test that bounding_box returns correct min and max coordinates."""
    nodes = np.array([[1, 2], [5, 3], [2, 7]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    min_coords, max_coords = shape.bounding_box()

    assert np.array_equal(min_coords, np.array([1, 2]))
    assert np.array_equal(max_coords, np.array([5, 7]))


def test_bounding_box_single_node():
    """Test bounding_box with a single node."""
    nodes = np.array([[3, 4]])
    edges = np.empty((0, 2, 2), dtype=int)
    shape = Shape(nodes=nodes, edges=edges)

    min_coords, max_coords = shape.bounding_box()

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
    min_coords, _ = originated.bounding_box()
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
    min_coords, _ = originated.bounding_box()
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
        min_coords, _ = rotated.bounding_box()
        assert np.array_equal(min_coords, np.array([0, 0])),             f"Rotation not at origin: min_coords = {min_coords}"


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
    edges2 = np.array([[[1, 0], [1, 1]], [[0, 0], [1, 0]]])  # Same edges, different order

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
            assert not unique_rots[i].equivalent(unique_rots[j]),                 f"Rotations {i} and {j} are equivalent but both in unique list"


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
            min_coords, max_coords = rotated.bounding_box()
            shape_width = max_coords[0] - min_coords[0] + 1
            shape_height = max_coords[1] - min_coords[1] + 1

            # Add 1 hex spacing between shapes
            base_offset = np.array([
                col * (shape_width + 1),
                row * (shape_height + 1)
            ])

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(fig, positioned.nodes, positioned.edges,
                      node_color=color, edge_color=color)

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
            min_c, max_c = rotated.bounding_box()
            width = max_c[0] - min_c[0] + 1
            height = max_c[1] - min_c[1] + 1
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    # Calculate grid size
    grid_width = shapes_per_row * rots_per_row * (max_width + 1) + 2
    grid_height = ((len(GIZMOS) + shapes_per_row - 1) // shapes_per_row) * 2 * (max_height + 1) + 2

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
            base_offset = np.array([
                shape_col * rots_per_row * (max_width + 1) + rot_col * (max_width + 1),
                shape_row * 2 * (max_height + 1) + rot_row * (max_height + 1)
            ])

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(fig, positioned.nodes, positioned.edges,
                      node_color=color, edge_color=color)

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
            min_c, max_c = rotated.bounding_box()
            width = max_c[0] - min_c[0] + 1
            height = max_c[1] - min_c[1] + 1
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    # Calculate grid size
    grid_width = shapes_per_row * rots_per_row * (max_width + 1) + 2
    grid_height = ((len(SPROCKETS) + shapes_per_row - 1) // shapes_per_row) * 2 * (max_height + 1) + 2

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
            base_offset = np.array([
                shape_col * rots_per_row * (max_width + 1) + rot_col * (max_width + 1),
                shape_row * 2 * (max_height + 1) + rot_row * (max_height + 1)
            ])

            # Translate the shape to its position
            positioned = rotated.translate(base_offset)

            # Use jittered color for variety
            color = positioned.jittered_color(jitter_amount=15)

            plot_shape(fig, positioned.nodes, positioned.edges,
                      node_color=color, edge_color=color)

    fig.show()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
