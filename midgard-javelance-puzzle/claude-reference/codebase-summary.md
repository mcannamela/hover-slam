# Javelance Codebase Summary

## What it is

A solver for a hexagonal-grid puzzle called **JAVELANCE**. The puzzle is a specific shaped region on a hex grid (a 21×11 "vertical box" with irregular cutouts), divided into 8 named sub-regions. The goal is to cover chosen regions using three types of pieces, at minimum total cost.

---

## Piece types

| Name | Nodes | Cost |
|------|-------|------|
| DOODAD | 4 | $1.00 |
| GIZMO | 3 | $5.40 |
| SPROCKET | 2 | $9.90 |

Each piece type has sub-variants (different edge/connection patterns). All pieces can be rotated to all 6 orientations on the hex grid. An uncovered target hex costs $14.30.

---

## Regions

The JAVELANCE shape is divided into 8 regions (indices 0–7):

| ID | Name |
|----|------|
| 0 | lower left |
| 1 | lower middle |
| 2 | lower right |
| 3 | right center |
| 4 | upper right |
| 5 | upper center right |
| 6 | upper center left |
| 7 | upper left |

---

## Module breakdown

### `shapes.py` — Core data structure
`Shape`: a set of hex-grid nodes (axial coordinates) plus edges (connections between adjacent hexes). Supports:
- Rotation (6 rotations via matrix multiplication on axial coords)
- Translation, union, difference, bounding box
- `unique_originated_rotations()` — deduplicates rotationally symmetric shapes
- LRU caching on most methods for performance

### `javelance.py` — Puzzle definition
Defines the puzzle geometry as `Shape` constants:
- `JAVELANCE` — the full playable region
- `JAVELANCE_REGIONS` — dict of 8 sub-regions
- `JAVELANCE_FORBIDDEN_EDGES` — edges pieces cannot cross (region boundaries become forbidden)
- `DOODADS`, `GIZMOS`, `SPROCKETS` — piece template lists

### `packing.py` — Solver
`PackingProblem`: target region + piece set + forbidden edges + optional overflow region.

`greedy_pack()`: greedy solver with pluggable heuristics. Two modes:
- `recompute_heuristic=False` (default): compute all priorities once, then greedily select — fast
- `recompute_heuristic=True`: recompute after each placement — adaptive, slower

**Heuristics** (registered in `HEURISTIC_REGISTRY`):

| Name | Description |
|------|-------------|
| `cost_per_node` | Prioritize lowest cost-per-node pieces |
| `largest_first` | Prioritize most nodes covered |
| `cheapest_first` | Prioritize cheapest piece |
| `expected_coverage_cost` | Prioritize efficient coverage of hard-to-cover nodes |
| `expected_coverage_cost_lookahead` | Same + 1-step lookahead for top-10 candidates |

`PackingSolution`: result containing placements, total cost, coverage fraction, covered/target node sets.

Cost formula: `sum(piece costs) + num_uncovered_target_nodes × 14.3`

### `schemas.py` — Validation
Pandera schema (`PackingResultsSchema`) for validating the summary results dataframe.

### `plotting.py` — Visualization
Plotly-based rendering of hex grids and packing solutions. Key functions:
- `plot_javelance()` — render the full puzzle with regions
- `plot_packing_solution()` — overlay placed pieces on puzzle

### `app.py` — Interactive UI
Dash web app for clicking hexes on the JAVELANCE grid to select/deselect them. Runs on port 8050.

### `main.py` — CLI
Three Typer commands:
- `pack-javelance` — run the solver across all C(8,4)=70 region combinations × 3 strategies, saving results as JSON + HTML plots + Parquet summary
- `analyze-solution` — annotate a saved solution JSON with rotation info and per-region piece counts
- `render-solution` — render a solution as overlayable PNG files (one per piece + composite)

---

## Optimal result from `output/2026-01-13T17-52-10`

**Best overall: Regions 0, 1, 3, 5 — `expected_coverage_cost` strategy**

| Metric | Value |
|--------|-------|
| Total cost | **$62.80** |
| Coverage | **100%** (all 86 target nodes covered) |
| DOODADs | 25 |
| GIZMOs | 7 |
| SPROCKETs | 0 |
| Total pieces | 32 |
| Empty target hexes | 0 |

Pieces touching each region (a piece straddling two regions is counted in both):

| Region | Name | Status | Pieces touching it |
|--------|------|--------|--------------------|
| 0 | lower left | TARGETED | 12 |
| 1 | lower middle | TARGETED | 15 |
| 2 | lower right | overflow | 6 |
| 3 | right center | TARGETED | 7 |
| 4 | upper right | overflow | 8 |
| 5 | upper center right | TARGETED | 13 |
| 6 | upper center left | overflow | 4 |
| 7 | upper left | overflow | 6 |

35 of the 128 placed nodes fell outside the targeted regions (overflow into adjacent non-target regions). No SPROCKETs were used — the solver found DOODADs and GIZMOs sufficient for full coverage at minimum cost.
