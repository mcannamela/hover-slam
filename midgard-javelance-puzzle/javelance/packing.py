"""Module for solving shape packing problems on hexagonal grids."""

import itertools
import math
import time
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from socket import send_fds
from typing import Callable, Self, Any

import numpy as np
from loguru import logger
from numpy import dtype, ndarray
from tqdm import tqdm

from javelance.shapes import Shape, union_shapes

# Type aliases for cleaner signatures
Candidate = tuple[str, Shape, float, int]  # (name, placement, cost, num_nodes)
PrioritizedCandidate = tuple[
    float, str, Shape, float, int
]  # (priority, name, placement, cost, num_nodes)

HeuristicFn = Callable[
    [
        "PackingProblem",  # problem
        set[Shape.Node],  # occupied_nodes
        list[Candidate],  # all_candidates
    ],
    np.ndarray,  # array of priority values (lower is better), aligned with all_candidates
]

SelectorFn = Callable[
    [list[PrioritizedCandidate]],  # valid_prioritized_candidates
    PrioritizedCandidate | None,  # chosen candidate or None to stop
]


def get_array_combinations(arr: np.ndarray, m: int) -> np.ndarray:
    """
    Enumerate all combinations of m elements from a 1D numpy array.

    Args:
        arr: 1D numpy array.
        m: Number of elements in each combination.

    Returns:
        2D numpy array where each row is a combination.
    """
    return np.array(list(itertools.combinations(arr, m)))


@dataclass
class PlacedShape:
    """A shape that has been placed at a specific location."""

    shape: Shape
    name: str
    cost: float

    def __post_init__(self):
        """Validate the placed shape."""
        # Ensure shape is originated (minimum coords at 0,0)
        if not np.array_equal(self.shape.nodes.min(axis=0), np.array([0, 0])):
            raise ValueError("PlacedShape must have originated shape")


@dataclass
class PackingProblem:
    """Definition of a shape packing problem."""

    target: Shape  # Nodes that must be covered
    pieces: list[tuple[str, Shape, float]]  # (name, shape, cost) for each piece type
    forbidden_edges: set[Shape.Edge]  # Edges that cannot be used
    allowed: Shape | None = None  # Additional nodes that may be covered (optional)
    uncovered_node_cost: float = 14.3  # Cost per uncovered target node

    def __post_init__(self):
        s = self.target.union(self.allowed) if self.allowed is not None else self.target
        self._admissible_edges = s.interior_edges() | s.boundary_edges()
        self._admissible_nodes = s.node_set()

    def admissible_nodes(self) -> set[tuple[int, int]]:
        return self._admissible_nodes

    def admissible_edges(self) -> set[tuple[int, int]]:
        return self._admissible_edges

    def is_valid_placement(self, shape: Shape, occupied_nodes: set[Shape.Node]) -> bool:
        """
        Check if a shape placement is valid.

        A placement is valid if:
        1. At least one node is in the target shape
        2. All nodes are within target ∪ allowed
        3. No nodes overlap with already occupied nodes
        4. No edges are in the forbidden edge set
        """
        shape_nodes = shape.node_set()
        admissible_edges = self.admissible_edges()

        admissible_nodes = self.admissible_nodes()

        target_nodes = self.target.node_set()

        # Check if at least one node is in target
        if not (shape_nodes & target_nodes):
            return False

        # Check if all nodes are in allowed region (target ∪ allowed)
        if not shape_nodes <= admissible_nodes:
            return False

        # Check for overlaps with occupied nodes
        if shape_nodes & occupied_nodes:
            return False

        # Check if any edges are forbidden
        shape_edges = shape.edge_set()
        if (
            bool(shape_edges & self.forbidden_edges)
            or not shape_edges <= admissible_edges
        ):
            return False

        return True

    def generate_all_placements(self, shape: Shape) -> list[Shape]:
        """
        Generate all valid placements of a shape within the target ∪ allowed region.

        Returns a list of translated shapes representing all valid placements.
        """
        valid_placements = []

        # Get the bounding box of the allowed region (target ∪ allowed)
        if self.allowed is not None:
            # Create union of target and allowed for bounding box calculation
            search_region = self.target.union(self.allowed)
        else:
            search_region = self.target

        search_min, search_max = search_region.bounding_addresses()

        # Get all unique rotations
        unique_rotations = shape.unique_originated_rotations()

        # For each rotation, try all possible translations
        for rotated in unique_rotations:
            # Try all positions where the shape could fit
            for i in range(search_min[0], search_max[0] + 2):
                for j in range(search_min[1], search_max[1] + 2):
                    offset = np.array([i, j])
                    translated = rotated.translate(offset)

                    # Quick bounds check
                    trans_min, trans_max = translated.bounding_addresses()
                    if (
                        trans_min[0] < search_min[0]
                        or trans_min[1] < search_min[1]
                        or trans_max[0] > search_max[0]
                        or trans_max[1] > search_max[1]
                    ):
                        continue

                    # Check if this is a valid placement
                    if self.is_valid_placement(translated, set()):
                        valid_placements.append(translated)

        return valid_placements


@dataclass
class PackingSolution:
    """A solution to a packing problem."""

    placements: list[tuple[str, Shape]]  # (name, placed_shape) for each placed piece
    total_cost: float
    covered_nodes: set[Shape.Node]
    coverage: float  # Fraction of target nodes covered
    target_nodes: set[Shape.Node]

    @classmethod
    def from_placements(
        cls,
        placements: list[tuple[str, Shape, float]],
        target_nodes: set[Shape.Node],
        uncovered_node_cost: float = 14.3,
    ) -> Self:
        """Create a PackingSolution from a list of placements."""
        placements_cost = sum(cost for _, _, cost in placements)
        covered = set()
        for _, shape, _ in placements:
            covered |= shape.node_set() & target_nodes
        n_uncovered = len(target_nodes - covered)
        total_cost = placements_cost + n_uncovered * uncovered_node_cost
        coverage = len(covered) / len(target_nodes) if target_nodes else 0

        placement_list = [(name, shape) for name, shape, _ in placements]

        return cls(
            placements=placement_list,
            total_cost=total_cost,
            covered_nodes=covered,
            coverage=coverage,
            target_nodes=target_nodes,
        )


# ============================================================================
# Heuristic Functions
# ============================================================================


def heuristic_cost_per_node(
    problem: PackingProblem,
    occupied_nodes: set[Shape.Node],
    all_candidates: list[Candidate],
) -> np.ndarray:
    """Prioritize pieces with lowest cost per node."""
    priorities = np.array(
        [cost / num_nodes for _, _, cost, num_nodes in all_candidates]
    )
    return priorities


def heuristic_largest_first(
    problem: PackingProblem,
    occupied_nodes: set[Shape.Node],
    all_candidates: list[Candidate],
) -> np.ndarray:
    """Prioritize largest pieces first."""
    priorities = np.array([-num_nodes for _, _, _, num_nodes in all_candidates])
    return priorities


def heuristic_cheapest_first(
    problem: PackingProblem,
    occupied_nodes: set[Shape.Node],
    all_candidates: list[Candidate],
) -> np.ndarray:
    """Prioritize cheapest pieces first."""
    priorities = np.array([cost for _, _, cost, _ in all_candidates])
    return priorities


def heuristic_expected_coverage_cost(
    problem: PackingProblem,
    occupied_nodes: set[Shape.Node],
    all_candidates: list[Candidate],
) -> np.ndarray:
    """
    Prioritize low-cost placements covering nodes with high expected coverage cost.

    For each node, calculates the expected cost to cover it based on all available
    placements that could cover it. Then prioritizes placements that efficiently
    cover high-cost nodes.
    """
    target_nodes = problem.target.node_set()

    # Calculate node expected coverage cost (only once for all candidates!)
    node_coverage_cost = {}
    for node in target_nodes:
        if node in occupied_nodes:
            # Already covered, skip
            continue

        # An uncovered node has some base cost
        total_cost = problem.uncovered_node_cost
        total_coverage = 1.0

        # Find all placements that cover this node
        for cand_name, cand_placement, cand_cost, cand_num_nodes in all_candidates:
            if node in cand_placement.node_set() - occupied_nodes:
                total_cost += cand_cost
                total_coverage += cand_num_nodes

        # Calculate expected coverage cost for this node
        node_coverage_cost[node] = (
            total_cost / total_coverage if total_coverage > 0 else 0
        )

    logger.debug(
        f"Most expensive node expected cost: {max(node_coverage_cost.values())} vs mean of {np.mean(list(node_coverage_cost.values()))}"
    )

    # Compute priorities for all candidates
    priorities = np.zeros(len(all_candidates))
    for idx, (name, placement, cost, num_nodes) in enumerate(all_candidates):
        # Sum of node expected coverage costs for all nodes in this placement
        expected_placement_coverage_costs = np.mean(
            [node_coverage_cost.get(node, 0) for node in placement.node_set()]
        )

        # Prioritize low-cost placements covering high-value nodes
        if expected_placement_coverage_costs > 0:
            priorities[idx] = cost / expected_placement_coverage_costs
        else:
            priorities[idx] = float("inf")  # No valuable nodes covered

    return priorities


def heuristic_expected_coverage_cost_lookahead(
    problem: PackingProblem,
    occupied_nodes: set[Shape.Node],
    all_candidates: list[Candidate],
    num_lookahead: int = 10,
) -> np.ndarray:
    """
    Enhanced heuristic with lookahead for top candidates.

    First computes priorities using expected_coverage_cost logic.
    Then for the best 10 candidates, performs lookahead:
    - Simulates placing each top candidate
    - Recomputes heuristic for remaining valid placements
    - Sets priority to: own_priority + min(recomputed_priorities)

    For candidates not in top 10, doubles their priority.
    """
    # Step 1: Compute initial priorities using expected_coverage_cost logic
    priorities = heuristic_expected_coverage_cost(
        problem, occupied_nodes, all_candidates
    )

    # Step 2: Identify top candidates (lowest priorities)
    # Get indices sorted by priority (ascending)
    sorted_indices = np.argsort(priorities)
    top_indices = sorted_indices[:num_lookahead]

    # Step 3: For each top candidate, perform lookahead
    lookahead_priorities = np.zeros(len(all_candidates))

    for idx in top_indices:
        if priorities[idx] == float("inf"):
            # Skip invalid candidates
            lookahead_priorities[idx] = float("inf")
            continue

        # Simulate placing this candidate
        _, placement, _, _ = all_candidates[idx]
        simulated_occupied = occupied_nodes | placement.node_set()

        # Filter remaining valid candidates (excluding this one and those that conflict)
        remaining_candidates = []
        for other_idx, (name, other_placement, cost, num_nodes) in enumerate(
            all_candidates
        ):
            if other_idx == idx:
                continue  # Skip the current candidate

            # Check if placement is still valid after simulated placement
            if problem.is_valid_placement(other_placement, simulated_occupied):
                remaining_candidates.append((name, other_placement, cost, num_nodes))

        # Recompute heuristic for remaining candidates
        if remaining_candidates:
            recomputed_priorities = heuristic_expected_coverage_cost(
                problem, simulated_occupied, remaining_candidates
            )
            min_recomputed = np.min(recomputed_priorities)
        else:
            # No remaining candidates, use 0 as the future cost
            min_recomputed = 0.0

        # Set lookahead priority: own priority + min future priority
        lookahead_priorities[idx] = priorities[idx] + min_recomputed

    # Step 4: For non-top candidates, double their priority
    non_top_indices = sorted_indices[num_lookahead:]
    for idx in non_top_indices:
        lookahead_priorities[idx] = priorities[idx] * 2.0

    return lookahead_priorities


# ============================================================================
# Selector Functions
# ============================================================================


def selector_min_priority(
    valid_prioritized_candidates: list[PrioritizedCandidate],
) -> PrioritizedCandidate | None:
    """Select the candidate with the lowest priority value (greedy selection)."""
    if not valid_prioritized_candidates:
        return None
    return min(valid_prioritized_candidates, key=lambda x: x[0])


# ============================================================================
# Strategy Registry
# ============================================================================

HEURISTIC_REGISTRY: dict[str, HeuristicFn] = {
    "cost_per_node": heuristic_cost_per_node,
    "largest_first": heuristic_largest_first,
    "cheapest_first": heuristic_cheapest_first,
    "expected_coverage_cost": heuristic_expected_coverage_cost,
    "expected_coverage_cost_lookahead": heuristic_expected_coverage_cost_lookahead,
}


@contextmanager
def log_elapsed(label="block"):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"Elapsed time for {label}: {(end - start):.2e}s")


def greedy_pack(
    problem: PackingProblem,
    initial_placements: list[tuple[str, Shape]] = None,
    strategy: str | None = None,
    heuristic_fn: HeuristicFn | None = None,
    selector_fn: SelectorFn | None = None,
    recompute_heuristic: bool = False,
) -> PackingSolution:
    """
    Pack shapes using a greedy algorithm with customizable heuristics.

    Args:
        problem: The packing problem to solve
        initial_placements: Initial placements in solution, defaults to empty set.
        strategy: String name of a registered strategy (e.g., "cost_per_node",
            "expected_coverage_cost"). If provided, overrides heuristic_fn.
        heuristic_fn: Custom heuristic function to compute placement priorities.
            If not provided and no strategy given, defaults to "cost_per_node".
        selector_fn: Function to select which placement to use from valid candidates.
            Defaults to selecting the minimum priority placement.
        recompute_heuristic: If True, recompute heuristic values after each placement.
            If False (default), compute all priorities once upfront. This is much more
            efficient but less adaptive to changing board states. Only use True for
            heuristics that need to adapt to occupied nodes.

    Returns:
        A PackingSolution with the greedy packing result
    """
    initial_placements = copy(initial_placements) or []
    if heuristic_kwargs is None:
        heuristic_kwargs = {}
    # Resolve the heuristic function
    if strategy is not None:
        if strategy not in HEURISTIC_REGISTRY:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(HEURISTIC_REGISTRY.keys())}"
            )
        heuristic_fn = HEURISTIC_REGISTRY[strategy]
    elif heuristic_fn is None:
        # Default to cost_per_node
        heuristic_fn = heuristic_cost_per_node

    # Default selector
    if selector_fn is None:
        selector_fn = selector_min_priority

    # Generate all possible placements for all pieces
    valid_candidates: list[Candidate] = []
    occupied_nodes = union_shapes([p for _, p, _ in initial_placements]).node_set()
    with log_elapsed("initialize_valid_placements"):
        for name, shape, cost in problem.pieces:
            piece_placements = problem.generate_all_placements(shape)
            for placement in piece_placements:
                if placement.node_set() & occupied_nodes:
                    continue
                num_nodes = len(placement.node_set())
                valid_candidates.append((name, placement, cost, num_nodes))
    if not valid_candidates:
        logger.info(f"No valid placements found for {problem}, solution is just the initial placements")
        return PackingSolution.from_placements(initial_placements, problem.target.node_set())
    if not recompute_heuristic:
        placements = _greedy_pack_once(
            initial_placements,
            heuristic_fn,
            problem,
            valid_candidates,
        )
    else:
        placements = _greedy_pack_iter(
            initial_placements,
            heuristic_fn,
            problem,
            selector_fn,
            valid_candidates,
        )
    target_nodes = problem.target.node_set()
    return PackingSolution.from_placements(placements, target_nodes)


def _greedy_pack_iter(
    placements: list[tuple[str, Shape]],
    heuristic_fn: Callable[
        [PackingProblem, set[tuple[int, int]], list[tuple[str, Shape, float, int]]],
        ndarray[tuple[Any, ...], dtype[Any]],
    ]
    | Callable[..., ndarray[tuple[Any, ...], dtype[Any]]],
    problem: PackingProblem,
    selector_fn: Callable[..., tuple[float, str, Shape, float, int] | None],
    valid_candidates: list[tuple[str, Shape, float, int]],
) -> list[tuple[str, Shape, float]]:
    placements: list[tuple[str, Shape, float]] = placements or []
    occupied_nodes = union_shapes(
        [placement for _, placement, _ in placements]
    ).node_set()

    # Check if there are any valid candidates left
    if not valid_candidates:
        logger.info("No valid candidates available, returning initial placements.")
        return placements

    # Recompute heuristic after each placement (adaptive but slower)
    n_target_nodes = len(problem.target.node_set())
    min_placement_size = min(
        len(placement.node_set()) for _, placement, _, _ in valid_candidates
    )
    max_iter = math.ceil(n_target_nodes / min_placement_size)
    logger.info(
        f"There are {n_target_nodes} target nodes and {len(valid_candidates)} candidate placements with min size {min_placement_size}: max_iter = {max_iter}."
    )
    for _ in tqdm(range(max_iter)):
        # Find all currently valid candidates and their indices
        valid_indices = [
            idx
            for idx, (name, placement, cost, num_nodes) in enumerate(valid_candidates)
            if problem.is_valid_placement(placement, occupied_nodes)
        ]

        if not valid_indices:
            # No more valid placements
            break

        # Compute priorities for all candidates (heuristic may need full context)
        priorities = heuristic_fn(
            problem, occupied_nodes, valid_candidates
        )

        # Extract valid prioritized candidates
        prioritized_candidates: list[PrioritizedCandidate] = [
            (priorities[idx], *valid_candidates[idx]) for idx in valid_indices
        ]

        # Select a placement using the selector function
        selected = selector_fn(prioritized_candidates)
        if selected is None:
            # Selector chose to stop
            break

        priority, name, placement, cost, num_nodes = selected

        # Place the selected piece
        placements.append((name, placement, cost))
        occupied_nodes |= placement.node_set()

        # filter newly invalid placements
        valid_candidates = [
            c
            for c in valid_candidates
            if problem.is_valid_placement(c[1], occupied_nodes)
        ]
    return placements


def _greedy_pack_once(
    placements: list[tuple[str, Shape]],
    heuristic_fn: Callable[
        [PackingProblem, set[tuple[int, int]], list[tuple[str, Shape, float, int]]],
        ndarray[tuple[Any, ...], dtype[Any]],
    ]
    | Callable[..., ndarray[tuple[Any, ...], dtype[Any]]],
    problem: PackingProblem,
    valid_candidates: list[tuple[str, Shape, float, int]],
) -> list[tuple[str, Shape, float]]:
    placements: list[tuple[str, Shape, float]] = placements or []
    occupied_nodes = union_shapes(
        [placement for _, placement, _ in placements]
    ).node_set()
    # Compute priorities once upfront for efficiency
    with log_elapsed("compute_priorities_once"):
        logger.info(f"There are {len(valid_candidates)} candidate placements.")
        priorities = heuristic_fn(
            problem, occupied_nodes, valid_candidates
        )

    # Zip priorities with candidates
    prioritized_all: list[PrioritizedCandidate] = [
        (priority, name, placement, cost, num_nodes)
        for priority, (name, placement, cost, num_nodes) in zip(
            priorities, valid_candidates
        )
    ]

    # Sort by priority once
    prioritized_all.sort(key=lambda x: x[0])

    # Greedy selection from pre-computed priorities
    with log_elapsed("greedy_select_once"):
        for priority, name, placement, cost, num_nodes in prioritized_all:
            if problem.is_valid_placement(placement, occupied_nodes):
                placements.append((name, placement, cost))
                occupied_nodes |= placement.node_set()
    return placements


# ============================================================================
# Iterative Improvement
# ============================================================================


def _shape_to_key(shape: Shape) -> tuple:
    """
    Create a hashable key from a Shape for use in dictionaries.

    Args:
        shape: Shape to convert to key

    Returns:
        Hashable tuple representing the shape
    """
    # Convert nodes to frozenset of tuples
    nodes_key = frozenset(tuple(node) for node in shape.nodes)

    # Convert edges to frozenset of tuples of tuples
    edges_key = frozenset(
        (tuple(edge[0]), tuple(edge[1])) for edge in shape.edges
    )

    return (nodes_key, edges_key)


@dataclass
class PlacementIntersections:
    """Data structures for analyzing placement overlaps and intersections."""

    # For each placement index, set of other placement indices it intersects with
    intersecting_placements: dict[int, set[int]]

    # For each pair of intersecting placements (i, j), the set of nodes in intersection
    # Both (i, j) and (j, i) are stored for easy lookup
    intersection_nodes: dict[tuple[int, int], set[Shape.Node]]

    # For each pair of intersecting placements (i, j), the count of nodes
    # Both (i, j) and (j, i) are stored for easy lookup
    intersection_counts: dict[tuple[int, int], int]

    # For each node, the set of placement indices that cover it
    node_to_placements: dict[Shape.Node, set[int]]

    # Map from shape to its index in the placements list
    shape_to_index: dict[tuple, int]


def build_placement_intersections(
    placements: list[tuple[str, Shape, float]]
) -> PlacementIntersections:
    """
    Build data structures for analyzing placement overlaps.

    Args:
        placements: List of (name, shape, cost) tuples

    Returns:
        PlacementIntersections with all intersection information
    """
    n = len(placements)

    intersecting_placements: dict[int, set[int]] = {i: set() for i in range(n)}
    intersection_nodes: dict[tuple[int, int], set[Shape.Node]] = {}
    intersection_counts: dict[tuple[int, int], int] = {}
    node_to_placements: dict[Shape.Node, set[int]] = {}
    shape_to_index: dict[tuple, int] = {}

    # Build node_to_placements and shape_to_index maps
    for idx, (name, shape, cost) in enumerate(placements):
        # Map shape to index
        shape_key = _shape_to_key(shape)
        shape_to_index[shape_key] = idx

        # Map nodes to placements
        for node in shape.node_set():
            if node not in node_to_placements:
                node_to_placements[node] = set()
            node_to_placements[node].add(idx)

    # Build intersection information
    for i in range(n):
        nodes_i = placements[i][1].node_set()

        for j in range(i + 1, n):
            nodes_j = placements[j][1].node_set()

            # Find intersection
            intersection = nodes_i & nodes_j

            if intersection:
                # Record that these placements intersect
                intersecting_placements[i].add(j)
                intersecting_placements[j].add(i)

                # Store intersection details for both orderings
                intersection_nodes[(i, j)] = intersection
                intersection_nodes[(j, i)] = intersection
                count = len(intersection)
                intersection_counts[(i, j)] = count
                intersection_counts[(j, i)] = count

    return PlacementIntersections(
        intersecting_placements=intersecting_placements,
        intersection_nodes=intersection_nodes,
        intersection_counts=intersection_counts,
        node_to_placements=node_to_placements,
        shape_to_index=shape_to_index,
    )


def iterative_improvement_pack(
    problem: PackingProblem,
    initial_placements: list[tuple[str, Shape, float]],
    max_iterations: int = 100,
) -> PackingSolution:
    """
    Iteratively improve a packing solution by finding better placements.

    This function takes an initial solution and attempts to improve it by:
    1. Identifying placements that overlap (share nodes)
    2. Attempting to replace overlapping placements with better alternatives
    3. Continuing until no improvements can be found

    Args:
        problem: The packing problem to solve
        initial_placements: Initial placements to start from (name, shape, cost)
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved PackingSolution
    """
    # Generate all possible valid placements for all piece types
    logger.info("Generating all possible placements...")
    all_placements: list[tuple[str, Shape, float]] = []

    with log_elapsed("generate_all_placements"):
        for name, shape, cost in problem.pieces:
            piece_placements = problem.generate_all_placements(shape)
            for placement in piece_placements:
                all_placements.append((name, placement, cost))

    logger.info(f"Generated {len(all_placements)} total possible placements")

    # Build intersection data structures on ALL possible placements
    logger.info("Building intersection data structures...")
    with log_elapsed("build_intersections"):
        intersections = build_placement_intersections(all_placements)

    logger.info(
        f"Found {len(intersections.intersection_nodes) // 2} pairs of intersecting placements"
    )

    # Convert initial placements to indices in all_placements
    current_solution_indices: set[int] = set()
    for name, shape, cost in initial_placements:
        shape_key = _shape_to_key(shape)
        if shape_key in intersections.shape_to_index:
            idx = intersections.shape_to_index[shape_key]
            current_solution_indices.add(idx)
        else:
            logger.warning(
                f"Initial placement shape not found in all_placements: {shape_key}"
            )

    logger.info(
        f"Starting iterative improvement with {len(current_solution_indices)} placements"
    )

    # TODO: Implement improvement logic
    # For now, just return the initial solution

    target_nodes = problem.target.node_set()
    return PackingSolution.from_placements(initial_placements, target_nodes)
