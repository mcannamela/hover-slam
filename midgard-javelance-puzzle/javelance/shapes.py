from dataclasses import dataclass
from typing import Self

import matplotlib.colors as mcolors
import numpy as np


# Rotation matrices for hex grid transformations
# Applied as coords @ matrix.T for rotations at 0°, 60°, 120°, 180°, 240°, 300°
ROTATION_MATRICES = [
    np.array([[1, 0], [0, 1]]),  # 0°
    np.array([[0, -1], [1, 1]]),  # 60°
    np.array([[-1, -1], [1, 0]]),  # 120°
    np.array([[-1, 0], [0, -1]]),  # 180°
    np.array([[0, 1], [-1, -1]]),  # 240°
    np.array([[1, 1], [-1, 0]]),  # 300°
]


@dataclass
class Shape:
    nodes: np.ndarray
    edges: np.ndarray
    mean_color: str = "rgb(128, 128, 128)"  # Default gray color

    def __post_init__(self):
        # Convert named colors to rgb format before validation
        color_str = self.mean_color.strip()
        if not (self._is_rgb_str(color_str) or self._is_rgba_str(color_str)):
            # Assume it's a named color and try to convert it
            try:
                # matplotlib's to_rgb returns tuple of floats in [0, 1] range
                rgb_tuple = mcolors.to_rgb(color_str)
                # Convert to 0-255 range
                r = int(rgb_tuple[0] * 255)
                g = int(rgb_tuple[1] * 255)
                b = int(rgb_tuple[2] * 255)
                # Set mean_color to rgb format
                self.mean_color = f"rgb({r}, {g}, {b})"
            except ValueError as e:
                # If conversion fails, let the validation method handle it
                pass

        self._raise_if_color_str_invalid()

        # ensure that nodes is a 2d array where dimension 1 has size 2
        if self.nodes.ndim != 2:
            raise ValueError(f"nodes must be a 2D array, got {self.nodes.ndim}D")
        if self.nodes.shape[1] != 2:
            raise ValueError(
                f"nodes must have size 2 in dimension 1, got {self.nodes.shape[1]}"
            )

        # ensure that nodes has integer type
        if not np.issubdtype(self.nodes.dtype, np.integer):
            raise ValueError(f"nodes must have integer type, got {self.nodes.dtype}")

        # ensure that edges is a 3d array where dimensions 1 and 2 both have size 2
        if self.edges.ndim != 3:
            raise ValueError(f"edges must be a 3D array, got {self.edges.ndim}D")
        if self.edges.shape[1] != 2:
            raise ValueError(
                f"edges must have size 2 in dimension 1, got {self.edges.shape[1]}"
            )
        if self.edges.shape[2] != 2:
            raise ValueError(
                f"edges must have size 2 in dimension 2, got {self.edges.shape[2]}"
            )

        # ensure that edges has integer type
        if not np.issubdtype(self.edges.dtype, np.integer):
            raise ValueError(f"edges must have integer type, got {self.edges.dtype}")

        # raise if any nodes or edges are duplicated
        nodes_set = set(map(tuple, self.nodes))
        if len(nodes_set) != len(self.nodes):
            raise ValueError("Duplicate nodes found")

        edges_set = set()
        for edge in self.edges:
            # Normalize edge representation (sort the two hexes to make comparison order-independent)
            edge_tuple = tuple(sorted([tuple(edge[0]), tuple(edge[1])]))
            if edge_tuple in edges_set:
                raise ValueError(f"Duplicate edge found: {edge}")
            edges_set.add(edge_tuple)

        # verify that all edges are valid i.e. the 2 hexes they address are adjacent
        # Valid adjacency offsets for pointy-top hexagons
        valid_offsets = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}

        for edge in self.edges:
            hex1 = edge[0]
            hex2 = edge[1]
            di = hex2[0] - hex1[0]
            dj = hex2[1] - hex1[1]
            offset = (di, dj)

            if offset not in valid_offsets:
                raise ValueError(
                    f"Invalid edge: hexagons {hex1} and {hex2} are not adjacent (offset {offset})"
                )

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the hexes whose coordinates are the lower and upper bounds of all nodes in the shape"""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def originated_rotations(self) -> list[Self]:
        """Return all rotations of the shape, but shifted such that all node coordinates are positive"""
        return [x.originated() for x in self.rotations()]

    def originated(self):
        """Shift the shape such that the minimum address for both coordinates is 0"""
        displacement = -np.min(self.nodes, axis=0, keepdims=True)
        return self.translate(displacement)

    def translate(self, displacement: np.ndarray) -> Self:
        """Shift the shape by the given displacement vector"""
        nodes_translated = self.nodes + displacement
        edges_translated = self.edges + displacement
        return self.__class__(
            nodes=nodes_translated, edges=edges_translated, mean_color=self.mean_color
        )

    def rotations(self) -> list[Self]:
        """
        Generate all 6 rotations of this shape on the hex grid.

        Returns a list of 6 Shape objects representing rotations by
        0°, 60°, 120°, 180°, 240°, and 300° counterclockwise.

        For axial coordinates (i, j), the rotation transformations are:
        - 0°:   (i, j) -> (i, j)
        - 60°:  (i, j) -> (-j, i+j)
        - 120°: (i, j) -> (-i-j, i)
        - 180°: (i, j) -> (-i, -j)
        - 240°: (i, j) -> (j, -i-j)
        - 300°: (i, j) -> (i+j, -i)
        """
        rotated_shapes = []

        for matrix in ROTATION_MATRICES:
            # Rotate all nodes using matrix multiplication
            rotated_nodes = self.nodes @ matrix.T

            # Rotate all edges
            if len(self.edges) > 0:
                # Reshape edges from (M, 2, 2) to (M*2, 2), rotate, then reshape back
                edges_reshaped = self.edges.reshape(-1, 2)
                rotated_edges_reshaped = edges_reshaped @ matrix.T
                rotated_edges = rotated_edges_reshaped.reshape(-1, 2, 2)
            else:
                rotated_edges = np.empty((0, 2, 2), dtype=int)

            rotated_shapes.append(
                Shape(
                    nodes=rotated_nodes, edges=rotated_edges, mean_color=self.mean_color
                )
            )

        return rotated_shapes

    def jittered_color(self, jitter_amount=20):
        """
        Generate a color by adding random noise to the mean_color in RGB space.

        Parameters:
        - jitter_amount: maximum amount to jitter each RGB channel (default 20)

        Returns:
        - A string in "rgb(r, g, b)" format with jittered values
        """
        r, g, b = self._parse_color()

        # Add random jitter to each channel
        r_jittered = r + np.random.randint(-jitter_amount, jitter_amount + 1)
        g_jittered = g + np.random.randint(-jitter_amount, jitter_amount + 1)
        b_jittered = b + np.random.randint(-jitter_amount, jitter_amount + 1)

        # Clamp values to valid range [0, 255]
        r_jittered = np.clip(r_jittered, 0, 255)
        g_jittered = np.clip(g_jittered, 0, 255)
        b_jittered = np.clip(b_jittered, 0, 255)

        return f"rgb({r_jittered}, {g_jittered}, {b_jittered})"

    def _parse_color(self) -> tuple[int, int, int]:
        # Parse the mean color to extract RGB values
        color_str = self.mean_color.strip()

        if self._is_rgb_str(color_str):
            # Extract RGB values from "rgb(r, g, b)" format
            rgb_str = color_str[4:-1]  # Remove "rgb(" and ")"
            r, g, b = map(int, rgb_str.split(","))
        elif self._is_rgba_str(color_str):
            # Extract RGB values from "rgba(r, g, b, a)" format
            rgba_str = color_str[5:-1]  # Remove "rgba(" and ")"
            r, g, b, _ = map(float, rgba_str.split(","))
            r, g, b = int(r), int(g), int(b)
        else:
            self._raise_if_color_str_invalid()

        return r, g, b

    def _is_rgba_str(self, color_str: str) -> bool:
        return color_str.startswith("rgba(") and color_str.endswith(")")

    def _is_rgb_str(self, color_str: str) -> bool:
        return color_str.startswith("rgb(") and color_str.endswith(")")

    def _raise_if_color_str_invalid(self):
        color_str = self.mean_color.strip()
        if not (self._is_rgb_str(color_str) or self._is_rgba_str(color_str)):
            # For named colors or other formats, default to a base gray
            # In practice, users should use rgb() format for mean_color if they want jittering
            raise ValueError(
                f"mean_color must be in 'rgb(r, g, b)' or 'rgba(r, g, b, a)' format, got '{color_str}'"
            )


DOODADS = [
    Shape(
        nodes=np.array(
            [
                [0, 0],
                [4, 0],
                [2, 1],  # in every derived shape
                [2, 4],
            ]
        ),
        edges=np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 0], [1, 1]],
                [[3, 0], [3, 1]],
                [[1, 2], [2, 2]],
                [[1, 3], [2, 2]],
                [[1, 3], [2, 3]],
                [[1, 4], [2, 3]],
            ]
        ),
        mean_color="chartreuse",
    )
]

GIZMOS = [
    Shape(
        nodes=np.array(
            [
                [4, 0],
                [2, 1],  # in every derived shape
                [2, 4],
            ]
        ),
        edges=np.array(
            [
                [[3, 0], [3, 1]],
                [[1, 2], [2, 2]],
                [[1, 3], [2, 2]],
                [[1, 3], [2, 3]],
                [[1, 4], [2, 3]],
            ]
        ),
        mean_color="blue",
    ),
    Shape(
        nodes=np.array(
            [
                [0, 0],
                [2, 1],  # in every derived shape
                [2, 4],
            ]
        ),
        edges=np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 0], [1, 1]],
                [[1, 2], [2, 2]],
                [[1, 3], [2, 2]],
                [[1, 3], [2, 3]],
                [[1, 4], [2, 3]],
            ]
        ),
        mean_color="yellow",
    ),
    Shape(
        nodes=np.array(
            [
                [0, 0],
                [4, 0],
                [2, 1],  # in every derived shape
            ]
        ),
        edges=np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 0], [1, 1]],
                [[3, 0], [3, 1]],
            ]
        ),
        mean_color="red",
    ),
]

SPROCKETS = [
    Shape(
        nodes=np.array(
            [
                [2, 1],  # in every derived shape
                [2, 4],
            ]
        ),
        edges=np.array(
            [
                [[1, 2], [2, 2]],
                [[1, 3], [2, 2]],
                [[1, 3], [2, 3]],
                [[1, 4], [2, 3]],
            ]
        ),
        mean_color="cyan",
    ),
    Shape(
        nodes=np.array(
            [
                [4, 0],
                [2, 1],  # in every derived shape
            ]
        ),
        edges=np.array(
            [
                [[3, 0], [3, 1]],
            ]
        ),
        mean_color="purple",
    ),
    Shape(
        nodes=np.array(
            [
                [0, 0],
                [2, 1],  # in every derived shape
            ]
        ),
        edges=np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 0], [1, 1]],
            ]
        ),
        mean_color="orange",
    ),
]
