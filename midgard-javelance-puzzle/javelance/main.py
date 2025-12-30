from javelance.packing import PackingProblem, greedy_pack
from javelance.plotting import plot_packing_solution
from javelance.shapes import (
    DOODADS,
    GIZMOS,
    JAVELANCE,
    JAVELANCE_FORBIDDEN_EDGES,
    SPROCKETS,
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
    print("\n=== Testing different greedy strategies ===")

    for strategy in ["cost_per_node", "largest_first", "cheapest_first"]:
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

        title = f"Strategy: {strategy} (Coverage: {solution.coverage:.2%})"
        fig = plot_packing_solution(solution, title=title)


if __name__ == "__main__":
    main()
