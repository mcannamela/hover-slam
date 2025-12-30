"""Tests for JAVELANCE regions and structures."""

import pytest

from javelance.javelance import JAVELANCE, JAVELANCE_REGIONS


def test_javelance_regions_are_disjoint():
    """Test that all JAVELANCE_REGIONS have disjoint nodesets."""
    # Get all region keys
    region_keys = sorted(JAVELANCE_REGIONS.keys())

    # Check all pairs of regions for disjoint nodesets
    for i, key_i in enumerate(region_keys):
        for key_j in region_keys[i + 1 :]:
            nodes_i = JAVELANCE_REGIONS[key_i].node_set()
            nodes_j = JAVELANCE_REGIONS[key_j].node_set()

            # Check that the intersection is empty
            intersection = nodes_i & nodes_j
            assert (
                len(intersection) == 0
            ), f"Regions {key_i} and {key_j} overlap with nodes: {intersection}"


def test_javelance_regions_union_equals_javelance():
    """Test that the union of all JAVELANCE_REGIONS equals JAVELANCE's nodeset."""
    # Compute the union of all region nodesets
    union_nodes = set()
    for region_key, region_shape in JAVELANCE_REGIONS.items():
        union_nodes |= region_shape.node_set()

    # Get JAVELANCE's nodeset
    javelance_nodes = JAVELANCE.node_set()

    # Check that they are equal
    assert union_nodes == javelance_nodes, (
        f"Union of regions ({len(union_nodes)} nodes) does not match JAVELANCE "
        f"({len(javelance_nodes)} nodes). "
        f"Missing from union: {javelance_nodes - union_nodes}. "
        f"Extra in union: {union_nodes - javelance_nodes}"
    )


def test_javelance_regions_coverage():
    """Test coverage statistics for JAVELANCE_REGIONS."""
    # Count total nodes
    total_nodes = 0
    for region_key, region_shape in JAVELANCE_REGIONS.items():
        num_nodes = len(region_shape.node_set())
        total_nodes += num_nodes
        print(f"Region {region_key}: {num_nodes} nodes")

    javelance_node_count = len(JAVELANCE.node_set())

    print(f"\nTotal nodes across regions: {total_nodes}")
    print(f"JAVELANCE total nodes: {javelance_node_count}")

    # This test just prints info but also verifies the count matches
    assert total_nodes == javelance_node_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
