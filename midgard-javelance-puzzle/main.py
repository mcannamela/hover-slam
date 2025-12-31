import functools

import numpy as np
from loguru import logger

from javelance.javelance import (
    DOODADS,
    GIZMOS,
    JAVELANCE,
    JAVELANCE_FORBIDDEN_EDGES,
    JAVELANCE_REGIONS,
    SPROCKETS,
)
from javelance.packing import PackingProblem, get_array_combinations, greedy_pack
from javelance.plotting import plot_packing_solution
from javelance.shapes import Shape, union_shapes


def main():
    # Set up the packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, 1.0))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, 5.4))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, 9.9))
    packing_results = []
    region_combinations = get_array_combinations(np.arange(len(JAVELANCE_REGIONS)), 4)

    for combo in region_combinations[:3]:
        logger.info(f"Testing regions: {combo}")
        regions = [JAVELANCE_REGIONS[i] for i in combo]
        target = union_shapes(regions)
        problem = PackingProblem(
            target=target,
            pieces=pieces,
            forbidden_edges=JAVELANCE_FORBIDDEN_EDGES,
            allowed=JAVELANCE.difference(target),
        )

        # Try different strategies
        logger.info("\n=== Testing different greedy strategies ===")

        for strategy, kwargs in [
            ("largest_first", {}),
            ("expected_coverage_cost", {"recompute_heuristic": True}),
        ]:
            solution = greedy_pack(problem, strategy=strategy, **kwargs)

            logger.info(f"\nStrategy: {strategy}")
            logger.info(f"  Coverage: {solution.coverage:.2%}")
            logger.info(f"  Total cost: {solution.total_cost:.2f}")
            logger.info(
                f"  Covered nodes: {len(solution.covered_nodes)}/{len(solution.target_nodes)}"
            )
            logger.info(f"  Pieces placed: {len(solution.placements)}")

            # Count piece types
            piece_counts = {}
            for name, _ in solution.placements:
                piece_counts[name] = piece_counts.get(name, 0) + 1

            logger.info(f"  Piece breakdown: {piece_counts}")

            title = f"Targeted Regions: {combo}  <br>Strategy: {strategy} <br>(Coverage: {solution.coverage:.2%})<br>(Cost: {solution.total_cost:.2f})"

            fig = plot_packing_solution(regions, solution, title=title)
            fig.show()


if __name__ == "__main__":
    main()
