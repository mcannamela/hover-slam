import functools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from tqdm import tqdm

from javelance.javelance import (
    DOODADS,
    GIZMOS,
    JAVELANCE,
    JAVELANCE_FORBIDDEN_EDGES,
    JAVELANCE_REGIONS,
    SPROCKETS,
)
from javelance.packing import (
    PackingProblem,
    PackingSolution,
    get_array_combinations,
    greedy_pack,
)
from javelance.plotting import plot_packing_solution
from javelance.schemas import PackingResultsSchema
from javelance.shapes import Shape, union_shapes

app = typer.Typer()


def serialize_solution_to_json(solution: PackingSolution) -> dict:
    """Convert a PackingSolution to a JSON-serializable dict."""
    # Serialize placements (name, Shape)
    placements_serialized = []
    for name, shape in solution.placements:
        placements_serialized.append(
            {
                "name": name,
                "shape": {
                    "nodes": shape.nodes.tolist(),
                    "edges": shape.edges.tolist(),
                },
            }
        )

    # Convert numpy integers to Python integers for JSON serialization
    covered_nodes_list = sorted(
        [[int(x) for x in node] for node in solution.covered_nodes]
    )
    target_nodes_list = sorted(
        [[int(x) for x in node] for node in solution.target_nodes]
    )

    return {
        "placements": placements_serialized,
        "total_cost": float(solution.total_cost),
        "covered_nodes": covered_nodes_list,
        "coverage": float(solution.coverage),
        "target_nodes": target_nodes_list,
    }


def deserialize_solution_from_json(data: dict) -> PackingSolution:
    """Reconstruct a PackingSolution from a JSON dict."""
    # Deserialize placements
    placements = []
    for p in data["placements"]:
        shape = Shape(
            nodes=np.array(p["shape"]["nodes"]),
            edges=np.array(p["shape"]["edges"]),
        )
        placements.append((p["name"], shape))

    # Deserialize node sets
    covered_nodes = {tuple(node) for node in data["covered_nodes"]}
    target_nodes = {tuple(node) for node in data["target_nodes"]}

    return PackingSolution(
        placements=placements,
        total_cost=data["total_cost"],
        covered_nodes=covered_nodes,
        coverage=data["coverage"],
        target_nodes=target_nodes,
    )


@app.command()
def main(
    show_plots: bool = typer.Option(
        False,
        "--show-plots/--no-show-plots",
        help="Whether to display plots interactively while solving",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        help="Random seed for shuffling the order of region combinations",
    ),
    max_combinations: int = typer.Option(
        None,
        "--max-combinations",
        help="Maximum number of region combinations to solve",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    sort_by_previous: str = typer.Option(
        None,
        "--sort-by-previous",
        help="Path to previous summary_results.parquet to sort combinations by cost",
    ),
):
    """
    Solve packing problems for different region combinations.

    By default, solves the first few combinations without showing plots.
    Use --show-plots to display interactive plots, --seed to shuffle combinations,
    and --max-combinations to limit how many are solved.

    Use --sort-by-previous to load a previous run's results and solve combinations
    ordered by their previous total cost (lowest cost first).
    """
    # Configure logging level
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Create output directory with timestamp
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_dir = Path(__file__).parent / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {output_dir}")

    # Set up the packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, 1.0))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, 5.4))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, 9.9))

    # Track all results for summary dataframe
    summary_results = []
    result_counter = 0

    # Get all region combinations
    region_combinations = get_array_combinations(np.arange(len(JAVELANCE_REGIONS)), 4)

    # Sort by previous results if provided
    if sort_by_previous is not None:
        if seed is not None:
            logger.warning(
                "Both --sort-by-previous and --seed provided; --sort-by-previous takes precedence"
            )

        # Load previous results
        previous_results_path = Path(sort_by_previous)
        if not previous_results_path.exists():
            raise FileNotFoundError(f"Previous results file not found: {sort_by_previous}")

        logger.info(f"Loading previous results from: {sort_by_previous}")
        previous_df = pl.read_parquet(previous_results_path)

        # Parse targeted_regions and compute mean cost per combination
        # Group by targeted_regions and take the minimum total_cost (best strategy)
        cost_by_regions = (
            previous_df.group_by("targeted_regions")
            .agg(pl.col("total_cost").min().alias("min_cost"))
            .sort("min_cost")
        )

        # Create a mapping from region tuple to cost
        region_costs = {}
        for row in cost_by_regions.iter_rows(named=True):
            region_str = row["targeted_regions"]
            region_tuple = tuple(sorted(map(int, region_str.split(","))))
            region_costs[region_tuple] = row["min_cost"]

        # Sort combinations: first those in previous results (by cost), then the rest
        combinations_with_cost = []
        combinations_without_cost = []

        for combo in region_combinations:
            combo_tuple = tuple(sorted(combo))
            if combo_tuple in region_costs:
                combinations_with_cost.append((region_costs[combo_tuple], combo))
            else:
                combinations_without_cost.append(combo)

        # Sort known combinations by cost
        combinations_with_cost.sort(key=lambda x: x[0])

        # Rebuild region_combinations: sorted known ones first, then unknown ones
        region_combinations = np.array(
            [combo for _, combo in combinations_with_cost] + combinations_without_cost
        )

        logger.info(
            f"Sorted {len(combinations_with_cost)} combinations by previous cost "
            f"({len(combinations_without_cost)} new combinations added at end)"
        )

    # Shuffle if seed is provided (and sort_by_previous not set)
    elif seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(region_combinations)
        logger.info(
            f"Shuffled {len(region_combinations)} combinations with seed={seed}"
        )

    # Limit to max_combinations if specified
    if max_combinations is not None:
        region_combinations = region_combinations[:max_combinations]
        logger.info(f"Limited to {max_combinations} combinations")
    else:
        # Default to all combinations if max not given
        region_combinations = region_combinations

    logger.info(f"Solving {len(region_combinations)} region combinations")

    for combo in tqdm(region_combinations):
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
            # ("largest_first", {}),
            ("expected_coverage_cost", {"recompute_heuristic": True}),
            # ("expected_coverage_cost_lookahead", {"recompute_heuristic": True}),
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

            # Collect summary data for dataframe
            num_target_nodes = len(solution.target_nodes)
            num_covered_target_nodes = len(solution.covered_nodes)
            num_empty_target_hexes = len(solution.target_nodes - solution.covered_nodes)

            cost_per_target_node = solution.total_cost / num_target_nodes

            n_doodads = piece_counts.get("DOODAD", 0)
            n_gizmods = piece_counts.get("GIZMO", 0)
            n_sprockets = piece_counts.get("SPROCKET", 0)

            # Create descriptive filename components
            regions_str = "-".join(map(str, combo))
            strategy_str = strategy.replace("_", "-")
            result_id = f"result_{result_counter:04d}"
            base_filename = f"{result_id}_regions_{regions_str}_strategy_{strategy_str}"

            # Save individual solution as JSON
            result_file = output_dir / f"{base_filename}_solution.json"
            with open(result_file, "w") as f:
                json.dump(serialize_solution_to_json(solution), f, indent=2)
            logger.info(f"  Saved solution to: {result_file}")

            # Generate plot
            title = (
                f"Targeted Regions: {combo}  <br>Strategy: {strategy}"
                f"<br>(Nodes, Covered, Uncovered)=({num_target_nodes},{num_covered_target_nodes}, {num_empty_target_hexes})"
                f"<br>(Doodads, Gizmos, Sprockets)=({n_doodads},{n_gizmods},{n_sprockets})"
                f"<br>Coverage: {solution.coverage:.2%}"
                f"<br>Cost: {solution.total_cost:.2f}, per node: {cost_per_target_node:.2f}"
            )
            fig = plot_packing_solution(regions, solution, title=title)

            # Show plot if requested
            if show_plots:
                fig.show()

            # Save plot as HTML
            plot_file = output_dir / f"{base_filename}_plot.html"
            fig.write_html(str(plot_file))
            logger.info(f"  Saved plot to: {plot_file}")

            summary_results.append(
                {
                    "targeted_regions": ",".join(map(str, combo)),
                    "packing_strategy": strategy,
                    "packing_strategy_params": json.dumps(kwargs),
                    "num_doodads": n_doodads,
                    "num_gizmos": n_gizmods,
                    "num_sprockets": n_sprockets,
                    "num_empty_target_hexes": num_empty_target_hexes,
                    "num_covered_target_nodes": len(solution.covered_nodes),
                    "num_target_nodes": num_target_nodes,
                    "total_cost": solution.total_cost,
                    "cost_per_target_node": cost_per_target_node,
                }
            )

            result_counter += 1

    # Create summary dataframe and validate with schema
    logger.info("\n=== Creating summary dataframe ===")
    df = pl.DataFrame(summary_results)
    validated_df = PackingResultsSchema.validate(df)
    logger.info(f"Created dataframe with {len(validated_df)} results")
    logger.info(f"\n{validated_df}")

    # Save summary dataframe
    summary_file = output_dir / "summary_results.parquet"
    validated_df.write_parquet(summary_file)
    logger.info(f"\nSaved summary to: {summary_file}")

    # Also save as CSV for easy viewing
    csv_file = output_dir / "summary_results.csv"
    validated_df.write_csv(csv_file)
    logger.info(f"Saved summary CSV to: {csv_file}")

    logger.info(f"\n=== All results saved to: {output_dir} ===")


if __name__ == "__main__":
    app()
