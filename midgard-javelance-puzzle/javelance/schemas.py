"""Pandera schemas for validating dataframes."""

import pandera.polars as pa


class PackingResultsSchema(pa.DataFrameModel):
    """Schema for packing problem solution results.

    This schema defines the structure for storing results from solving
    PackingProblem instances, particularly for JAVELANCE packing with
    DOODADS, GIZMOS, and SPROCKETS.
    """

    # Target regions being packed
    targeted_regions: str = pa.Field(
        description="Comma-separated list of targeted JAVELANCE region IDs (e.g., '0,1,2')"
    )

    # Packing strategy information
    packing_strategy: str = pa.Field(
        description="Name of the packing strategy used to generate this solution"
    )

    packing_strategy_params: str = pa.Field(
        description="Parameters used for the packing strategy (e.g., JSON string)"
    )

    # Piece counts
    num_doodads: int = pa.Field(
        ge=0,
        description="Number of DOODAD pieces placed in the solution"
    )

    num_gizmos: int = pa.Field(
        ge=0,
        description="Number of GIZMO pieces placed in the solution"
    )

    num_sprockets: int = pa.Field(
        ge=0,
        description="Number of SPROCKET pieces placed in the solution"
    )

    # Coverage metrics
    num_empty_target_hexes: int = pa.Field(
        ge=0,
        description="Number of target hexagons not covered by any piece"
    )

    num_covered_target_nodes: int = pa.Field(
        ge=0,
        description="Number of target nodes covered by placed pieces"
    )

    # Cost metric
    total_cost: float = pa.Field(
        ge=0.0,
        description="Total cost of the packing solution"
    )

    class Config:
        """Pandera config."""
        strict = True  # Disallow columns not in schema
        coerce = True  # Coerce data types to match schema
