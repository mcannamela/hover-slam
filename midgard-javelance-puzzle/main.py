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
def pack_javelance(
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
    doodad_cost: float = typer.Option(
        1.0,
        "--doodad-cost",
        help="Cost per DOODAD piece",
    ),
    gizmo_cost: float = typer.Option(
        5.4,
        "--gizmo-cost",
        help="Cost per GIZMO piece",
    ),
    sprocket_cost: float = typer.Option(
        9.9,
        "--sprocket-cost",
        help="Cost per SPROCKET piece",
    ),
    empty_hex_cost: float = typer.Option(
        14.3,
        "--empty-hex-cost",
        help="Cost per uncovered target node (empty hex)",
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

    # Log costs to file
    costs_info = {
        "doodad_cost": doodad_cost,
        "gizmo_cost": gizmo_cost,
        "sprocket_cost": sprocket_cost,
        "empty_hex_cost": empty_hex_cost,
    }
    costs_file = output_dir / "costs.json"
    with open(costs_file, "w") as f:
        json.dump(costs_info, f, indent=2)
    logger.info(
        f"Costs: DOODAD={doodad_cost}, GIZMO={gizmo_cost}, SPROCKET={sprocket_cost}, EMPTY_HEX={empty_hex_cost}"
    )
    logger.info(f"Saved costs to: {costs_file}")

    # Set up the packing problem
    pieces = []
    for doodad in DOODADS:
        pieces.append(("DOODAD", doodad, doodad_cost))
    for gizmo in GIZMOS:
        pieces.append(("GIZMO", gizmo, gizmo_cost))
    for sprocket in SPROCKETS:
        pieces.append(("SPROCKET", sprocket, sprocket_cost))

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
        previous_results_path = Path(__file__).parent / sort_by_previous
        if not previous_results_path.exists():
            raise FileNotFoundError(
                f"Previous results file not found: {sort_by_previous}"
            )

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
            # Add empty_hex_cost to heuristic_kwargs
            heuristic_kwargs = {"uncovered_node_cost": empty_hex_cost}
            solution = greedy_pack(
                problem, strategy=strategy, heuristic_kwargs=heuristic_kwargs, **kwargs
            )

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


@app.command()
def analyze_solution(
    solution_file: str = typer.Argument(
        ..., help="Path to solution JSON file to analyze"
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Output file for annotated solution (JSON format)"
    ),
):
    """
    Analyze a solution file and annotate each placement with its rotation.

    Loads a solution from a JSON file and determines which rotation of which
    piece template each placement corresponds to. Displays the annotated
    information and optionally saves it to a file.
    """
    solution_path = Path(solution_file)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_file}")

    logger.info(f"Loading solution from: {solution_file}")

    # Load the solution
    with open(solution_path, "r") as f:
        solution_data = json.load(f)
    solution = deserialize_solution_from_json(solution_data)

    # Build piece templates
    piece_templates = {
        "DOODAD": DOODADS,
        "GIZMO": GIZMOS,
        "SPROCKET": SPROCKETS,
    }

    logger.info(f"Analyzing {len(solution.placements)} placements...")

    # Annotate each placement
    annotated_placements = []

    for idx, (piece_name, placed_shape) in enumerate(solution.placements):
        # Find which template and rotation matches this placement
        template_idx, rotation_idx = find_template_and_rotation(
            placed_shape, piece_templates[piece_name]
        )

        annotation = {
            "placement_index": idx,
            "piece_type": piece_name,
            "template_index": template_idx,
            "rotation_index": rotation_idx,
            "rotation_degrees": rotation_idx * 60,
            "num_nodes": len(placed_shape.nodes),
            "nodes": placed_shape.nodes.tolist(),
        }

        annotated_placements.append(annotation)

        logger.info(
            f"  Placement {idx}: {piece_name}[{template_idx}] "
            f"rotation {rotation_idx} ({rotation_idx * 60}°)"
        )

    # Create annotated solution
    annotated_solution = {
        "original_solution_file": str(solution_path),
        "num_placements": len(solution.placements),
        "total_cost": solution.total_cost,
        "coverage": solution.coverage,
        "annotated_placements": annotated_placements,
    }

    # Display summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total placements: {len(solution.placements)}")
    logger.info(f"Total cost: {solution.total_cost:.2f}")
    logger.info(f"Coverage: {solution.coverage:.2%}")

    # Count by piece type
    piece_counts = {}
    for p in annotated_placements:
        piece_type = p["piece_type"]
        piece_counts[piece_type] = piece_counts.get(piece_type, 0) + 1

    logger.info(f"\nPiece counts:")
    for piece_type, count in sorted(piece_counts.items()):
        logger.info(f"  {piece_type}: {count}")

    # Count by rotation
    rotation_counts = {}
    for p in annotated_placements:
        rotation = p["rotation_degrees"]
        rotation_counts[rotation] = rotation_counts.get(rotation, 0) + 1

    logger.info(f"\nRotation distribution:")
    for rotation, count in sorted(rotation_counts.items()):
        logger.info(f"  {rotation}°: {count}")

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(annotated_solution, f, indent=2)
        logger.info(f"\nSaved annotated solution to: {output_file}")
    else:
        logger.info(f"\nUse --output to save annotated solution to a file")


def find_template_and_rotation(
    placed_shape: Shape, templates: list[Shape]
) -> tuple[int, int]:
    """
    Find which template and rotation matches the placed shape.

    Args:
        placed_shape: The shape from the solution placement
        templates: List of template shapes to compare against

    Returns:
        Tuple of (template_index, rotation_index)
    """
    placed_nodes = placed_shape.node_set()

    for template_idx, template in enumerate(templates):
        # Try each rotation of this template
        rotations = template.rotations()
        for rotation_idx, rotated in enumerate(rotations):
            # Check if this rotation matches the placed shape
            # We need to check if they have the same nodes (after translation)
            rotated_nodes = rotated.node_set()

            # Find translation offset
            if len(placed_nodes) != len(rotated_nodes):
                continue

            # Get first node from each to compute offset
            if not placed_nodes or not rotated_nodes:
                continue

            placed_first = min(placed_nodes)
            rotated_first = min(rotated_nodes)
            offset = (
                placed_first[0] - rotated_first[0],
                placed_first[1] - rotated_first[1],
            )

            # Translate rotated nodes by offset
            translated_nodes = {
                (node[0] + offset[0], node[1] + offset[1]) for node in rotated_nodes
            }

            # Check if they match
            if translated_nodes == placed_nodes:
                return (template_idx, rotation_idx)

    # If no match found, return (-1, -1)
    logger.warning(f"Could not find matching template and rotation for placed shape")
    return (-1, -1)


if __name__ == "__main__":
    app()
