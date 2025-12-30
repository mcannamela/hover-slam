"""Tests for pandera schemas."""

import polars as pl
import pytest

from javelance.schemas import PackingResultsSchema


def test_packing_results_schema_valid():
    """Test that PackingResultsSchema validates a valid dataframe."""
    # Create a valid dataframe
    df = pl.DataFrame({
        "targeted_regions": ["0,1,2"],
        "packing_strategy": ["greedy"],
        "packing_strategy_params": ["{}"],
        "num_doodads": [5],
        "num_gizmos": [3],
        "num_sprockets": [2],
        "num_empty_target_hexes": [10],
        "num_covered_target_nodes": [50],
        "total_cost": [123.45],
    })

    # Validate the dataframe
    validated_df = PackingResultsSchema.validate(df)

    # Check that validation succeeded
    assert validated_df is not None
    assert len(validated_df) == 1


def test_packing_results_schema_multiple_rows():
    """Test that PackingResultsSchema validates multiple rows."""
    # Create a dataframe with multiple solutions
    df = pl.DataFrame({
        "targeted_regions": ["0,1", "2,3,4", "5,6,7"],
        "packing_strategy": ["greedy", "random", "optimal"],
        "packing_strategy_params": ["{}", '{"seed": 42}', '{"max_time": 60}'],
        "num_doodads": [5, 3, 7],
        "num_gizmos": [3, 2, 4],
        "num_sprockets": [2, 1, 3],
        "num_empty_target_hexes": [10, 15, 5],
        "num_covered_target_nodes": [50, 45, 60],
        "total_cost": [123.45, 98.76, 150.00],
    })

    # Validate the dataframe
    validated_df = PackingResultsSchema.validate(df)

    # Check that validation succeeded
    assert validated_df is not None
    assert len(validated_df) == 3


def test_packing_results_schema_coercion():
    """Test that PackingResultsSchema coerces types correctly."""
    # Create a dataframe with coercible types
    df = pl.DataFrame({
        "targeted_regions": ["0"],
        "packing_strategy": ["greedy"],
        "packing_strategy_params": ["{}"],
        "num_doodads": [5.0],  # Float that should be coerced to int
        "num_gizmos": [3.0],
        "num_sprockets": [2.0],
        "num_empty_target_hexes": [10.0],
        "num_covered_target_nodes": [50.0],
        "total_cost": [123],  # Int that should be coerced to float
    })

    # Validate the dataframe
    validated_df = PackingResultsSchema.validate(df)

    # Check that types were coerced correctly
    assert validated_df["num_doodads"].dtype == pl.Int64
    assert validated_df["total_cost"].dtype == pl.Float64


def test_packing_results_schema_negative_values():
    """Test that PackingResultsSchema rejects negative values."""
    # Create a dataframe with negative values (should fail validation)
    df = pl.DataFrame({
        "targeted_regions": ["0"],
        "packing_strategy": ["greedy"],
        "packing_strategy_params": ["{}"],
        "num_doodads": [-1],  # Invalid: negative
        "num_gizmos": [3],
        "num_sprockets": [2],
        "num_empty_target_hexes": [10],
        "num_covered_target_nodes": [50],
        "total_cost": [123.45],
    })

    # Validation should raise an error
    with pytest.raises(Exception):  # Pandera raises schema errors
        PackingResultsSchema.validate(df)


def test_packing_results_schema_missing_column():
    """Test that PackingResultsSchema rejects dataframes with missing columns."""
    # Create a dataframe missing a required column
    df = pl.DataFrame({
        "targeted_regions": ["0"],
        "packing_strategy": ["greedy"],
        "packing_strategy_params": ["{}"],
        "num_doodads": [5],
        "num_gizmos": [3],
        # Missing num_sprockets
        "num_empty_target_hexes": [10],
        "num_covered_target_nodes": [50],
        "total_cost": [123.45],
    })

    # Validation should raise an error
    with pytest.raises(Exception):  # Pandera raises schema errors
        PackingResultsSchema.validate(df)


def test_packing_results_schema_extra_column():
    """Test that strict mode rejects dataframes with extra columns."""
    # Create a dataframe with an extra column
    df = pl.DataFrame({
        "targeted_regions": ["0"],
        "packing_strategy": ["greedy"],
        "packing_strategy_params": ["{}"],
        "num_doodads": [5],
        "num_gizmos": [3],
        "num_sprockets": [2],
        "num_empty_target_hexes": [10],
        "num_covered_target_nodes": [50],
        "total_cost": [123.45],
        "extra_column": ["should fail"],  # Extra column
    })

    # Validation should raise an error in strict mode
    with pytest.raises(Exception):  # Pandera raises schema errors
        PackingResultsSchema.validate(df)


def test_packing_results_schema_zero_values():
    """Test that PackingResultsSchema accepts zero values."""
    # Create a dataframe with zero values (should be valid)
    df = pl.DataFrame({
        "targeted_regions": [""],
        "packing_strategy": ["none"],
        "packing_strategy_params": [""],
        "num_doodads": [0],
        "num_gizmos": [0],
        "num_sprockets": [0],
        "num_empty_target_hexes": [0],
        "num_covered_target_nodes": [0],
        "total_cost": [0.0],
    })

    # Validate the dataframe
    validated_df = PackingResultsSchema.validate(df)

    # Check that validation succeeded
    assert validated_df is not None
    assert len(validated_df) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
