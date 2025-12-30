from loguru import logger

from javelance.packing import PackingProblem, greedy_pack
from javelance.plotting import plot_packing_solution
from javelance.javelance import (
    DOODADS,
    GIZMOS,
    SPROCKETS,
    JAVELANCE,
    JAVELANCE_FORBIDDEN_EDGES,
)


def main():
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
            f"  Covered nodes: {len(solution.covered_nodes)}/{len(JAVELANCE.node_set())}"
        )
        logger.info(f"  Pieces placed: {len(solution.placements)}")

        # Count piece types
        piece_counts = {}
        for name, _ in solution.placements:
            piece_counts[name] = piece_counts.get(name, 0) + 1

        logger.info(f"  Piece breakdown: {piece_counts}")

        title = f"Strategy: {strategy} (Coverage: {solution.coverage:.2%})"
        fig = plot_packing_solution(solution, title=title)
        fig.show()


if __name__ == "__main__":
    main()
