"""Tests for the packing module."""

import numpy as np
import pytest
from plotly.graph_objs import Figure

from javelance.javelance import JAVELANCE_REGIONS
from javelance.packing import (
    PackingProblem,
    PackingSolution,
    greedy_pack,
    get_array_combinations,
)
from javelance.plotting import plot_javelance, plot_packing_solution
from javelance.shapes import Shape, union_shapes


def test_get_array_combinations():
    """Test get_array_combinations function."""
    arr = np.array([1, 2, 3])
    m = 2
    expected = np.array([[1, 2], [1, 3], [2, 3]])
    result = get_array_combinations(arr, m)
    np.testing.assert_array_equal(result, expected)

    # Test with different m
    m = 3
    expected = np.array([[1, 2, 3]])
    result = get_array_combinations(arr, m)
    np.testing.assert_array_equal(result, expected)

    # Test with larger array
    arr = np.array([10, 20, 30, 40])
    m = 2
    result = get_array_combinations(arr, m)
    assert result.shape == (6, 2)


def test_is_valid_placement_all_nodes_in_target():
    """Test that is_valid_placement checks nodes are in target."""
    # Create a 3x3 box as target
    target = Shape.box(width=3, height=3)

    # Create a piece that fits
    piece = Shape.box(width=2, height=2)

    problem = PackingProblem(target=target, pieces=[], forbidden_edges=set())

    # Should be valid
    assert problem.is_valid_placement(piece, set())

    # Create a piece that goes outside target
    piece_outside = piece.translate(np.array([5, 5]))
    assert not problem.is_valid_placement(piece_outside, set())


def test_is_valid_placement_no_overlap():
    """Test that is_valid_placement checks for node overlap."""
    target = Shape.box(width=5, height=5)
    piece = Shape.box(width=2, height=2)

    problem = PackingProblem(target=target, pieces=[], forbidden_edges=set())

    # First placement should be valid
    assert problem.is_valid_placement(piece, set())

    # Mark some nodes as occupied
    occupied = {(0, 0), (1, 0)}

    # Now the placement should be invalid
    assert not problem.is_valid_placement(piece, occupied)

    # But a different position should be valid
    piece_offset = piece.translate(np.array([3, 3]))
    assert problem.is_valid_placement(piece_offset, occupied)


def test_is_valid_placement_forbidden_edges():
    """Test that is_valid_placement checks forbidden edges."""
    # Create a simple line of 3 nodes
    nodes = np.array([[0, 0], [1, 0], [2, 0]])
    edges = np.array([[[0, 0], [1, 0]], [[1, 0], [2, 0]]])
    target = Shape(nodes=nodes, edges=edges)

    # Create a piece with an edge
    piece_nodes = np.array([[0, 0], [1, 0]])
    piece_edges = np.array([[[0, 0], [1, 0]]])
    piece = Shape(nodes=piece_nodes, edges=piece_edges)

    # No forbidden edges - should be valid
    problem = PackingProblem(target=target, pieces=[], forbidden_edges=set())
    assert problem.is_valid_placement(piece, set())

    # Add the edge to forbidden list
    forbidden = {((0, 0), (1, 0))}
    problem_with_forbidden = PackingProblem(
        target=target, pieces=[], forbidden_edges=forbidden
    )
    assert not problem_with_forbidden.is_valid_placement(piece, set())


def test_is_valid_placement_with_allowed_region():
    """Test that is_valid_placement works with allowed region."""
    # Create a 3x3 target
    target = Shape.box(width=3, height=3)

    # Create a 2x2 allowed region adjacent to target
    allowed = Shape.box(width=2, height=2).translate(np.array([3, 0]))

    # Create a piece that spans both target and allowed
    piece = Shape.box(width=2, height=2).translate(np.array([2, 0]))

    problem = PackingProblem(
        target=target, pieces=[], forbidden_edges=set(), allowed=allowed
    )

    # Should be valid - has nodes in target and all nodes in target ∪ allowed
    assert problem.is_valid_placement(piece, set())


def test_is_valid_placement_all_nodes_in_allowed_only():
    """Test that placement with all nodes in allowed (but not target) is invalid."""
    # Create a 3x3 target
    target = Shape.box(width=3, height=3)

    # Create a 2x2 allowed region adjacent to target
    allowed = Shape.box(width=2, height=2).translate(np.array([3, 0]))

    # Create a piece entirely in allowed region (not touching target)
    piece = Shape.box(width=2, height=2).translate(np.array([3, 0]))

    problem = PackingProblem(
        target=target, pieces=[], forbidden_edges=set(), allowed=allowed
    )

    # Should be invalid - no nodes in target
    assert not problem.is_valid_placement(piece, set())


def test_is_valid_placement_nodes_outside_target_and_allowed():
    """Test that placement with nodes outside target ∪ allowed is invalid."""
    # Create a 3x3 target
    target = Shape.box(width=3, height=3)

    # Create a 2x2 allowed region adjacent to target
    allowed = Shape.box(width=2, height=2).translate(np.array([3, 0]))

    # Create a piece that goes outside both target and allowed
    piece = Shape.box(width=2, height=2).translate(np.array([10, 10]))

    problem = PackingProblem(
        target=target, pieces=[], forbidden_edges=set(), allowed=allowed
    )

    # Should be invalid - nodes outside target ∪ allowed
    assert not problem.is_valid_placement(piece, set())


def test_is_valid_placement_backward_compatibility():
    """Test that allowed=None behaves like the old implementation."""
    # Create a 3x3 box as target
    target = Shape.box(width=3, height=3)

    # Create a piece that fits
    piece = Shape.box(width=2, height=2)

    # Problem with allowed=None (backward compatible)
    problem = PackingProblem(
        target=target, pieces=[], forbidden_edges=set(), allowed=None
    )

    # Should be valid
    assert problem.is_valid_placement(piece, set())

    # Create a piece that goes outside target
    piece_outside = piece.translate(np.array([5, 5]))
    assert not problem.is_valid_placement(piece_outside, set())


def test_generate_all_placements_with_allowed_region():
    """Test generating placements with an allowed region."""
    # Create a small 2x2 target
    target = Shape.box(width=2, height=2)

    # Create a 2x2 allowed region adjacent to target
    allowed = Shape.box(width=2, height=2).translate(np.array([2, 0]))

    # Create a 2x2 piece
    piece = Shape.box(width=2, height=2)

    problem = PackingProblem(
        target=target, pieces=[], forbidden_edges=set(), allowed=allowed
    )

    placements = problem.generate_all_placements(piece)

    # Should find placements that:
    # 1. Fit entirely in target (1 placement at origin)
    # 2. Span target and allowed (1 placement at [1, 0])
    # But NOT placements entirely in allowed (would have no target nodes)

    # Verify all placements have at least one node in target
    target_nodes = target.node_set()
    for placement in placements:
        placement_nodes = placement.node_set()
        assert placement_nodes & target_nodes, (
            f"Placement {placement_nodes} has no target nodes"
        )

    # Should have at least 2 placements (one at origin, one spanning)
    assert len(placements) >= 2


def test_generate_all_placements_simple():
    """Test generating all placements for a simple piece."""
    # Create a 4x4 target
    target = Shape.box(width=4, height=4)

    # Create a 2x2 piece
    piece = Shape.box(width=2, height=2)

    problem = PackingProblem(target=target, pieces=[], forbidden_edges=set())

    placements = problem.generate_all_placements(piece)

    # For a 2x2 piece in a 4x4 target, there should be 3x3 = 9 valid positions
    # (positions (0,0), (0,1), (0,2), (1,0), ... (2,2))
    assert len(placements) >= 9

    # Check that all placements are valid
    for placement in placements:
        assert problem.is_valid_placement(placement, set())


def test_greedy_pack_perfect_fit():
    """Test greedy packing when piece perfectly fits target."""
    # Create a 2x2 target
    target = Shape.box(width=2, height=2)

    # Create a 2x2 piece
    piece = Shape.box(width=2, height=2)

    pieces = [("box_2x2", piece, 1.0)]
    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    solution = greedy_pack(problem)

    # Should have 100% coverage
    assert solution.coverage == 1.0

    # Should have exactly one placement
    assert len(solution.placements) == 1

    # Should cost 1.0
    assert solution.total_cost == 1.0


def test_greedy_pack_multiple_pieces():
    """Test greedy packing with multiple pieces."""
    # Create a 4x2 target (4 wide, 2 tall)
    target = Shape.box(width=4, height=2)

    # Create 2x2 pieces
    piece = Shape.box(width=2, height=2)

    # We should be able to fit 2 pieces
    pieces = [("box_2x2", piece, 1.0)]
    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    solution = greedy_pack(problem)

    # Should have 100% coverage
    assert solution.coverage == 1.0

    # Should have exactly two placements
    assert len(solution.placements) == 2

    # Should cost 2.0
    assert solution.total_cost == 2.0


def test_greedy_pack_partial_coverage():
    """Test greedy packing when full coverage is impossible."""
    # Create a 3x3 target
    target = Shape.box(width=3, height=3)

    # Create a 2x2 piece - can't perfectly tile a 3x3
    piece = Shape.box(width=2, height=2)

    pieces = [("box_2x2", piece, 1.0)]
    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    solution = greedy_pack(problem)

    # Coverage should be partial (not 0, not 1)
    assert 0 < solution.coverage < 1.0

    # Should have at least one placement
    assert len(solution.placements) >= 1


def test_greedy_pack_with_different_costs():
    """Test that greedy packing considers costs."""
    # Create a simple target
    target = Shape.box(width=4, height=4)

    # Create two piece types: cheap large vs expensive small
    large_piece = Shape.box(width=2, height=2)  # 4 nodes
    small_piece = Shape.box(width=1, height=1)  # 1 node

    # Strategy: cost_per_node should prefer large_piece (1.0/4 = 0.25) over small_piece (1.0/1 = 1.0)
    pieces = [
        ("large", large_piece, 1.0),
        ("small", small_piece, 1.0),
    ]

    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    solution = greedy_pack(problem, strategy="cost_per_node")

    # Should primarily use large pieces
    large_count = sum(1 for name, _ in solution.placements if name == "large")
    small_count = sum(1 for name, _ in solution.placements if name == "small")

    # Should have used more large pieces
    assert large_count > 0


def test_greedy_pack_line_tiling():
    """Test packing a line with smaller line segments."""
    # Create a horizontal line of 6 nodes
    nodes = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    edges = np.array(
        [
            [[0, 0], [1, 0]],
            [[1, 0], [2, 0]],
            [[2, 0], [3, 0]],
            [[3, 0], [4, 0]],
            [[4, 0], [5, 0]],
        ]
    )
    target = Shape(nodes=nodes, edges=edges)

    # Create a piece that is a line of 2 nodes
    piece_nodes = np.array([[0, 0], [1, 0]])
    piece_edges = np.array([[[0, 0], [1, 0]]])
    piece = Shape(nodes=piece_nodes, edges=piece_edges)

    pieces = [("line_2", piece, 1.0)]
    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    solution = greedy_pack(problem)

    # Should achieve 100% coverage (6 nodes / 2 nodes per piece = 3 pieces)
    assert solution.coverage == 1.0
    assert len(solution.placements) == 3
    assert solution.total_cost == 3.0


def test_packing_solution_from_placements():
    """Test creating a PackingSolution from placements."""
    piece = Shape.box(width=2, height=2)
    # A 2x2 box has 4 nodes
    target_nodes = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)}

    placements = [
        ("piece1", piece, 1.5),
        ("piece2", piece.translate(np.array([2, 0])), 1.5),
    ]

    solution = PackingSolution.from_placements(placements, target_nodes)

    assert solution.total_cost == 3.0
    # Two 2x2 boxes cover 8 nodes
    assert len(solution.covered_nodes) == 8
    assert solution.coverage == 1.0


def test_javelance_packing():
    """Test packing DOODADS, GIZMOS, and SPROCKETS onto JAVELANCE."""
    from javelance.javelance import JAVELANCE_FORBIDDEN_EDGES
    from javelance.javelance import JAVELANCE
    from javelance.javelance import SPROCKETS
    from javelance.javelance import GIZMOS
    from javelance.javelance import DOODADS

    # Set up the packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, 1.0))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, 5.4))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, 9.9))

    problem = PackingProblem(
        target=JAVELANCE, pieces=pieces, forbidden_edges=JAVELANCE_FORBIDDEN_EDGES
    )

    # Try different strategies
    print("\n=== Testing different greedy strategies ===")

    for strategy in [
        "cost_per_node",
        "largest_first",
        "cheapest_first",
        "expected_coverage_cost",
    ]:
        solution = greedy_pack(problem, strategy=strategy)

        print(f"\nStrategy: {strategy}")
        print(f"  Coverage: {solution.coverage:.2%}")
        print(f"  Total cost: {solution.total_cost:.2f}")
        print(
            f"  Covered nodes: {len(solution.covered_nodes)}/{len(JAVELANCE.node_set())}"
        )
        print(f"  Pieces placed: {len(solution.placements)}")

        # Count piece types
        piece_counts = {}
        for name, _ in solution.placements:
            piece_counts[name] = piece_counts.get(name, 0) + 1

        print(f"  Piece breakdown: {piece_counts}")

    # The test passes if we get some coverage
    assert solution.coverage > 0


def test_heuristic_expected_coverage_cost_performance():
    """Test and benchmark the expected_coverage_cost heuristic."""
    import time
    from javelance.packing import heuristic_expected_coverage_cost

    # Create a simple test problem
    target = Shape.box(width=10, height=10)  # 100 nodes
    piece = Shape.box(width=2, height=2)  # 4 nodes

    pieces = [("test_piece", piece, 1.0)]
    problem = PackingProblem(target=target, pieces=pieces, forbidden_edges=set())

    # Generate all candidates
    all_candidates = []
    for name, shape, cost in problem.pieces:
        piece_placements = problem.generate_all_placements(shape)
        for placement in piece_placements:
            num_nodes = len(placement.node_set())
            all_candidates.append((name, placement, cost, num_nodes))

    print(f"\n=== Heuristic Performance Test ===")
    print(f"Target size: {len(target.node_set())} nodes")
    print(f"Total candidates: {len(all_candidates)}")

    # Time a single heuristic call (now computes all priorities at once)
    occupied_nodes = set()

    start = time.time()
    priorities = heuristic_expected_coverage_cost(
        problem, occupied_nodes, all_candidates
    )
    elapsed = time.time() - start

    print(
        f"Single heuristic call (all {len(all_candidates)} candidates): {elapsed:.4f}s"
    )
    print(f"First priority value: {priorities[0]:.6f}")
    print(f"Time per candidate: {elapsed / len(all_candidates) * 1000:.2f}ms")

    # The heuristic should compute all priorities efficiently
    assert elapsed < 1.0, (
        f"Heuristic too slow: {elapsed:.4f}s for all {len(all_candidates)} candidates"
    )


def test_heuristic_expected_coverage_cost_javelance_size():
    """Test heuristic performance with JAVELANCE-sized problem."""
    import time
    from javelance.packing import heuristic_expected_coverage_cost
    from javelance.javelance import JAVELANCE_FORBIDDEN_EDGES
    from javelance.javelance import JAVELANCE
    from javelance.javelance import SPROCKETS
    from javelance.javelance import GIZMOS
    from javelance.javelance import DOODADS

    # Set up the actual JAVELANCE packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, 1.0))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, 5.4))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, 9.9))

    problem = PackingProblem(
        target=JAVELANCE, pieces=pieces, forbidden_edges=JAVELANCE_FORBIDDEN_EDGES
    )

    # Generate all candidates (this is what greedy_pack does)
    print(f"\n=== JAVELANCE Heuristic Performance ===")
    print(f"Target size: {len(JAVELANCE.node_set())} nodes")

    start_gen = time.time()
    all_candidates = []
    for name, shape, cost in problem.pieces:
        piece_placements = problem.generate_all_placements(shape)
        for placement in piece_placements:
            num_nodes = len(placement.node_set())
            all_candidates.append((name, placement, cost, num_nodes))
    elapsed_gen = time.time() - start_gen

    print(f"Candidate generation: {elapsed_gen:.2f}s")
    print(f"Total candidates: {len(all_candidates)}")

    # Test a single heuristic call (now computes all priorities at once!)
    occupied_nodes = set()

    start = time.time()
    priorities = heuristic_expected_coverage_cost(
        problem, occupied_nodes, all_candidates
    )
    elapsed = time.time() - start

    print(
        f"Single heuristic call (all {len(all_candidates)} candidates): {elapsed:.4f}s"
    )
    print(f"First priority value: {priorities[0]:.6f}")
    print(f"Time per candidate: {elapsed / len(all_candidates) * 1000:.2f}ms")

    # Calculate complexity
    target_size = len(JAVELANCE.node_set())
    num_candidates = len(all_candidates)

    # The heuristic does: for each target node, check all candidates ONCE
    # New complexity: O(target_nodes * candidates) instead of O(target_nodes * candidates^2)
    node_set_calls = target_size * num_candidates
    print(
        f"Node set calls total: {node_set_calls:,} ({target_size} * {num_candidates})"
    )
    print(f"New complexity: O({node_set_calls:,}) = {target_size} * {num_candidates}")
    print(
        f"Previous complexity would have been: O({target_size * num_candidates * num_candidates:,})"
    )


def test_javelance_packing_visualization():
    """Visualize the JAVELANCE packing solution."""
    from javelance.javelance import JAVELANCE_FORBIDDEN_EDGES
    from javelance.javelance import SPROCKETS
    from javelance.javelance import GIZMOS
    from javelance.javelance import DOODADS

    # Set up the packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, 1.0))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, 5.4))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, 9.9))

    combo = range(4)
    regions = [JAVELANCE_REGIONS[i] for i in combo]
    problem = PackingProblem(
        target=union_shapes(regions),
        pieces=pieces,
        forbidden_edges=JAVELANCE_FORBIDDEN_EDGES,
    )

    strategy = "cost_per_node"
    solution = greedy_pack(problem, strategy=strategy)

    print(f"\n=== Best Solution (cost_per_node) ===")
    print(f"Coverage: {solution.coverage:.2%}")
    print(f"Total cost: {solution.total_cost:.2f}")
    print(f"Pieces placed: {len(solution.placements)}")

    title = f"Targeted Regions: {combo}  <br>Strategy: {strategy} <br>(Coverage: {solution.coverage:.2%})<br>(Cost: {solution.total_cost:.2f})"
    fig = plot_packing_solution(regions, solution, title=title)

    fig.show()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
