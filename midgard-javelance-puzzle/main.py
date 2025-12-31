import functools
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
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
from javelance.schemas import PackingResultsSchema
from javelance.shapes import Shape, union_shapes


def main():
    # Create output directory with timestamp
    timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    output_dir = Path("output") / timestamp
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

            # Generate plot
            title = f"Targeted Regions: {combo}  <br>Strategy: {strategy} <br>(Coverage: {solution.coverage:.2%})<br>(Cost: {solution.total_cost:.2f})"
            fig = plot_packing_solution(regions, solution, title=title)
            fig.show()

            # Save individual solution
            result_id = f"result_{result_counter:04d}"
            result_file = output_dir / f"{result_id}_solution.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(solution, f)
            logger.info(f"  Saved solution to: {result_file}")

            # Save plot as HTML
            plot_file = output_dir / f"{result_id}_plot.html"
            fig.write_html(str(plot_file))
            logger.info(f"  Saved plot to: {plot_file}")

            # Collect summary data for dataframe
            num_empty_target_hexes = len(solution.target_nodes - solution.covered_nodes)
            summary_results.append({
                "targeted_regions": ",".join(map(str, combo)),
                "packing_strategy": strategy,
                "packing_strategy_params": json.dumps(kwargs),
                "num_doodads": piece_counts.get("DOODAD", 0),
                "num_gizmos": piece_counts.get("GIZMO", 0),
                "num_sprockets": piece_counts.get("SPROCKET", 0),
                "num_empty_target_hexes": num_empty_target_hexes,
                "num_covered_target_nodes": len(solution.covered_nodes),
                "total_cost": solution.total_cost,
            })

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
    main()
