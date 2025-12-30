import functools
import operator
from dataclasses import dataclass
from typing import Self, Any
from plotly.graph_objs import Figure
import matplotlib.colors as mcolors
import numpy as np
from numpy import dtype, ndarray

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

ADJACENCY_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
ADJACENCY_OFFSETS_ARRAY = np.array(ADJACENCY_OFFSETS, dtype=int)


@dataclass
class Shape:
    nodes: np.ndarray
    edges: np.ndarray
    mean_color: str = "rgb(128, 128, 128)"  # Default gray color

    Node = tuple[int, int]
    Edge = tuple[Node, Node]

    @classmethod
    def adjacent_nodes(cls, nodes: np.ndarray) -> np.ndarray:
        """Return the adjacent nodes of the given nodes"""
        # add singleton dimensions as needed to `nodes` and `ADJACENCY_OFFSETS_ARRAY`, then add the two
        # finally, reshape to obtain a valid array of nodes that is n_adjacent x 2
        # nodes shape: (N, 2) -> (N, 1, 2)
        # ADJACENCY_OFFSETS_ARRAY shape: (6, 2) -> (1, 6, 2)
        # Broadcasting gives: (N, 6, 2)
        # Reshape to: (N*6, 2)
        if len(nodes) == 0:
            return np.empty((0, 2), dtype=int)

        nodes_expanded = nodes[:, np.newaxis, :]  # (N, 1, 2)
        offsets_expanded = ADJACENCY_OFFSETS_ARRAY[np.newaxis, :, :]  # (1, 6, 2)
        adjacent = nodes_expanded + offsets_expanded  # (N, 6, 2)
        return adjacent.reshape(-1, 2)  # (N*6, 2)

    @classmethod
    def vertical_box(cls, width: int, height: int, mean_color: str = None) -> Self:
        row_shape = (width, 1)
        row_nodes = np.concatenate(
            [
                np.arange(width, dtype=int)[:, np.newaxis],
                np.zeros(row_shape, dtype=int),
            ],
            axis=1,
        )
        rows = []
        for i in range(height):
            vertical_offset = i
            horzontal_offset = -(i // 2)
            offset = np.array([horzontal_offset, vertical_offset], dtype=int)
            rows.append(row_nodes + offset)
        nodes = np.concatenate(rows, axis=0)
        if mean_color is None:
            mean_color = "rgb(128, 128, 128)"
        return cls(nodes=nodes, edges=Shape.empty_edges(), mean_color=mean_color)

    @classmethod
    def box(cls, width: int, height: int, mean_color: str = None) -> Self:
        row_shape = (width, 1)
        row_nodes = np.concatenate(
            [
                np.arange(width, dtype=int)[:, np.newaxis],
                np.zeros(row_shape, dtype=int),
            ],
            axis=1,
        )
        rows = []
        for i in range(height):
            offset = np.array([0, i], dtype=int)
            rows.append(row_nodes + offset)
        nodes = np.concatenate(rows, axis=0)
        if mean_color is None:
            mean_color = "rgb(128, 128, 128)"
        return cls(nodes=nodes, edges=Shape.empty_edges(), mean_color=mean_color)

    @classmethod
    def empty_edges(cls) -> ndarray[tuple[int, int, int], dtype[int]]:
        return np.empty((0, 2, 2), dtype=int)

    @classmethod
    def normalize_edge(cls, edge) -> Edge:
        """Normalize an edge by sorting its nodes to ensure consistent comparison."""
        return tuple(sorted([tuple(edge[0]), tuple(edge[1])]))

    @classmethod
    def as_node_set(cls, nodes: np.ndarray) -> set[Node]:
        return set(map(tuple, nodes))

    @classmethod
    def as_edge_set(cls, edges: np.ndarray) -> set[Edge]:
        return set(cls.normalize_edge(edge) for edge in edges)

    @classmethod
    def from_sets(
        cls,
        nodes: set[Node] = frozenset(),
        edges: set[Edge] = frozenset(),
        mean_color: str = None,
    ) -> Self:
        """Construct a Shape from sets of nodes and edges."""
        # Convert nodes set to numpy array
        if len(nodes) > 0:
            nodes_array = np.array(sorted(nodes), dtype=int)
        else:
            nodes_array = np.empty((0, 2), dtype=int)

        # Convert edges set to numpy array
        if len(edges) > 0:
            edges_list = [np.array([list(edge[0]), list(edge[1])]) for edge in edges]
            edges_array = np.array(edges_list, dtype=int)
        else:
            edges_array = cls.empty_edges()

        # Use default color if None
        if mean_color is None:
            mean_color = "rgb(128, 128, 128)"

        return cls(nodes=nodes_array, edges=edges_array, mean_color=mean_color)

    def adjacent(self) -> Self:
        """Return a Shape with all the adjacent nodes of this Shape's nodes but no edges"""
        # Get all adjacent nodes (may have duplicates)
        adjacent_nodes = self.adjacent_nodes(self.nodes)
        # Convert to set to remove duplicates, then back to array
        unique_nodes = np.array(sorted(set(map(tuple, adjacent_nodes))), dtype=int)
        return self.__class__(
            nodes=unique_nodes,
            edges=Shape.empty_edges(),
            mean_color=self.mean_color,
        )

    def negative_nodes(self) -> Self:
        """Return a Shape with all the nodes in this Shape's bounding_box that are not in the Shape and no edges"""
        return self.bounding_box().difference(self)

    def width(self):
        """Number of columns spanned by the shape"""
        return self.size()[0]

    def height(self):
        """Number of rows spanned by the shape"""
        return self.size()[1]

    def size(self) -> np.ndarray:
        """Difference between min and max coordinates of the shape"""
        return (
            self.bounding_box().bounding_addresses()[1]
            - self.bounding_box().bounding_addresses()[0]
        )

    def bounding_box(self) -> Self:
        """A Shape that contains all the nodes between the bounding addresses of this Shape"""
        min_address, max_address = self.bounding_addresses()
        width = max_address[0] - min_address[0] + 1
        height = max_address[1] - min_address[1] + 1
        return Shape.box(
            width=width, height=height, mean_color=self.mean_color
        ).translate(min_address)

    def difference(self, other: Self) -> Self:
        """The shape whose node and edge sets are the set difference of this shape and the other shape's sets."""
        nodes = self.node_set() - other.node_set()
        edges = self.edge_set() - other.edge_set()
        return Shape.from_sets(nodes=nodes, edges=edges, mean_color=self.mean_color)

    def node_set(self) -> set[Node]:
        """This Shape's nodes as a set"""
        return self.as_node_set(self.nodes)

    def edge_set(self) -> set[Edge]:
        """This Shape's edges as a set, normalized so that the nodes comprising the edge are ordered"""
        return self.as_edge_set(self.edges)

    def interior_edges(self) -> set[Edge]:
        """Set of all valid edges that can be made from this shape's nodes"""
        nodes = self.node_set()
        adjacent_nodes_in_shape = {
            n: self.as_node_set(self.adjacent_nodes(np.array([n]))) & nodes
            for n in nodes
        }
        edges = set()
        for n, adj in adjacent_nodes_in_shape.items():
            these_edges = {self.normalize_edge((n, a)) for a in adj}
            edges |= these_edges

        return edges

    def boundary_edges(self) -> set[Edge]:
        """Set of all edges that are not interior edges but have one node in the shape"""
        nodes = self.node_set()
        adjacent_nodes_out_of_shape = {
            n: self.as_node_set(self.adjacent_nodes(np.array([n]))) - nodes
            for n in nodes
        }
        edges = set()
        for n, adj in adjacent_nodes_out_of_shape.items():
            these_edges = {self.normalize_edge((n, a)) for a in adj}
            edges |= these_edges

        return edges

    def bounding_addresses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the hexes whose coordinates are the lower and upper bounds of all nodes in the shape"""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def originated_rotations(self) -> list[Self]:
        """Return all rotations of the shape, but shifted such that all node coordinates are positive"""
        return [x.originated() for x in self.rotations()]

    def equivalent(self, other: Self) -> bool:
        """
        Check if two shapes are equivalent (same nodes and edges after origination).

        Two shapes are considered equivalent if their originated versions have
        the same set of nodes and the same set of edges.
        """
        # Origin both shapes
        self_originated = self.originated()
        other_originated = other.originated()

        # Check if nodes are the same (as sets, order doesn't matter)
        self_nodes_set = set(map(tuple, self_originated.nodes))
        other_nodes_set = set(map(tuple, other_originated.nodes))

        if self_nodes_set != other_nodes_set:
            return False

        # Check if edges are the same (as sets, order doesn't matter)
        # Normalize each edge by sorting its two hexes
        self_edges_set = set(
            self.normalize_edge(edge) for edge in self_originated.edges
        )
        other_edges_set = set(
            self.normalize_edge(edge) for edge in other_originated.edges
        )

        return self_edges_set == other_edges_set

    def unique_originated_rotations(self) -> list[Self]:
        """
        Return only unique rotations of the shape (after origination).

        Shapes with rotational symmetry may have fewer than 6 unique rotations.
        This method filters out duplicates using the equivalent() method.
        """
        originated_rots = self.originated_rotations()
        unique_rots = []

        for rot in originated_rots:
            # Check if this rotation is equivalent to any already found
            is_duplicate = False
            for unique_rot in unique_rots:
                if rot.equivalent(unique_rot):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_rots.append(rot)

        return unique_rots

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
                rotated_edges = Shape.empty_edges()

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
            edge_tuple = self.normalize_edge(edge)
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
        mean_color="DarkGreen",
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
        mean_color="DarkGoldenRod",
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

JAVELANCE_COLOR = "Maroon"
JAVELANCE_FORBIDDEN_COLOR = "LightGrey"
JAVELANCE_VBOX = Shape.vertical_box(width=21, height=11, mean_color=JAVELANCE_COLOR)
JAVELANCE_GRID_SHAPE = Shape.vertical_box(
    width=23, height=13, mean_color=JAVELANCE_COLOR
)
JAVELANCE_FORBIDDEN_NODES = functools.reduce(
    operator.or_,
    [
        {(x, y) for x in xx}
        for d in [
            {
                0: [
                    0,
                    1,
                    2,
                    3,
                    7,
                    8,
                    10,
                    11,
                    19,
                    20,
                ]
            },
            {1: [0, 1, 2, 7]},
            {2: [-1, 0, 1, 13]},
            {
                3: [
                    -1,
                    1,
                    12,
                ]
            },
            {4: [-2, -1, 5, 17, 18]},
            {5: [-2, 5, 16, 18]},
            {6: [-3]},
            {7: [5, 17]},
            {8: [-1, -4, 4, 5, 13]},
            {9: [-4, -2, -1, 4, 13, 16]},
            {10: [-5, -4, -3, -2, -1, 3, 4, 7]},
        ]
        for y, xx in d.items()
    ],
    set(),
)

JAVELANCE = JAVELANCE_VBOX.difference(
    Shape.from_sets(nodes=JAVELANCE_FORBIDDEN_NODES, edges=set())
)

JAVELANCE_FORBIDDEN_EDGES = JAVELANCE.adjacent().difference(JAVELANCE).interior_edges()

JAVELANCE_FORBIDDEN = Shape.from_sets(
    nodes=JAVELANCE_FORBIDDEN_NODES,
    edges=JAVELANCE_FORBIDDEN_EDGES,
    mean_color=JAVELANCE_FORBIDDEN_COLOR,
)
