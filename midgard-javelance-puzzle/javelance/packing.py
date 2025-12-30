"""Module for solving shape packing problems on hexagonal grids."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from javelance.shapes import Shape


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

    target: Shape  # The shape to pack into
    pieces: list[tuple[str, Shape, float]]  # (name, shape, cost) for each piece type
    forbidden_edges: set[Shape.Edge]  # Edges that cannot be used

    def is_valid_placement(self, shape: Shape, occupied_nodes: set[Shape.Node]) -> bool:
        """
        Check if a shape placement is valid.

        A placement is valid if:
        1. All nodes are within the target shape
        2. No nodes overlap with already occupied nodes
        3. No edges are in the forbidden edge set
        """
        shape_nodes = shape.node_set()
        target_nodes = self.target.node_set()

        # Check if all nodes are in target
        if not shape_nodes <= target_nodes:
            return False

        # Check for overlaps with occupied nodes
        if shape_nodes & occupied_nodes:
            return False

        # Check if any edges are forbidden
        shape_edges = shape.edge_set()
        if shape_edges & self.forbidden_edges:
            return False

        return True

    def generate_all_placements(self, shape: Shape) -> list[Shape]:
        """
        Generate all valid placements of a shape within the target.

        Returns a list of translated shapes representing all valid placements.
        """
        valid_placements = []

        # Get the bounding box of the target
        target_min, target_max = self.target.bounding_addresses()

        # Get all unique rotations
        unique_rotations = shape.unique_originated_rotations()

        # For each rotation, try all possible translations
        for rotated in unique_rotations:
            rot_min, rot_max = rotated.bounding_addresses()
            shape_width = rot_max[0] - rot_min[0]
            shape_height = rot_max[1] - rot_min[1]

            # Try all positions where the shape could fit
            for i in range(target_min[0], target_max[0] + 2):
                for j in range(target_min[1], target_max[1] + 2):
                    offset = np.array([i, j])
                    translated = rotated.translate(offset)

                    # Quick bounds check
                    trans_min, trans_max = translated.bounding_addresses()
                    if (
                        trans_min[0] < target_min[0]
                        or trans_min[1] < target_min[1]
                        or trans_max[0] > target_max[0]
                        or trans_max[1] > target_max[1]
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
            covered |= shape.node_set()
        n_uncovered = len(target_nodes - covered)
        total_cost = placements_cost + n_uncovered * uncovered_node_cost
        coverage = len(covered) / len(target_nodes) if target_nodes else 0

        placement_list = [(name, shape) for name, shape, _ in placements]

        return cls(
            placements=placement_list,
            total_cost=total_cost,
            covered_nodes=covered,
            coverage=coverage,
        )


def greedy_pack(
    problem: PackingProblem, strategy: str = "cost_per_node"
) -> PackingSolution:
    """
    Pack shapes using a greedy algorithm.

    Args:
        problem: The packing problem to solve
        strategy: Greedy strategy to use:
            - "cost_per_node": Prioritize pieces with lowest cost per node
            - "largest_first": Prioritize largest pieces first
            - "cheapest_first": Prioritize cheapest pieces first
            - "expected_coverage_cost": Prioritize low-cost placements covering
              nodes with high expected coverage cost

    Returns:
        A PackingSolution with the greedy packing result
    """
    occupied_nodes = set()
    placements = []
    target_nodes = problem.target.node_set()

    # Generate all possible placements for all pieces
    all_candidates = []
    for name, shape, cost in problem.pieces:
        piece_placements = problem.generate_all_placements(shape)
        for placement in piece_placements:
            num_nodes = len(placement.node_set())
            all_candidates.append((name, placement, cost, num_nodes))

    # Calculate node expected coverage cost if using that strategy
    if strategy == "expected_coverage_cost":
        node_coverage_cost = {}
        for node in target_nodes:
            total_cost = 14.3
            total_coverage = 1
            # Find all placements that cover this node
            for name, placement, cost, num_nodes in all_candidates:
                if node in placement.node_set():
                    total_cost += cost
                    total_coverage += num_nodes

            # Calculate expected coverage cost for this node
            if total_coverage > 0:
                node_coverage_cost[node] = total_cost / total_coverage
            else:
                node_coverage_cost[node] = 0

    # Calculate priority for each candidate
    prioritized_candidates = []
    for name, placement, cost, num_nodes in all_candidates:
        if strategy == "cost_per_node":
            priority = cost / num_nodes
        elif strategy == "largest_first":
            priority = -num_nodes
        elif strategy == "cheapest_first":
            priority = cost
        elif strategy == "expected_coverage_cost":
            # Sum of node expected coverage costs for all nodes in this placement
            total_node_value = sum(
                node_coverage_cost.get(node, 0) for node in placement.node_set()
            )
            # Prioritize low-cost placements covering high-value nodes
            # Lower priority value = better
            if total_node_value > 0:
                priority = cost / total_node_value
            else:
                priority = float("inf")  # No valuable nodes covered
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        prioritized_candidates.append((priority, name, placement, cost, num_nodes))

    # Sort by priority (lower is better)
    prioritized_candidates.sort(key=lambda x: x[0])

    # Greedily place shapes
    for priority, name, placement, cost, num_nodes in prioritized_candidates:
        if problem.is_valid_placement(placement, occupied_nodes):
            placements.append((name, placement, cost))
            occupied_nodes |= placement.node_set()

    return PackingSolution.from_placements(placements, target_nodes)
