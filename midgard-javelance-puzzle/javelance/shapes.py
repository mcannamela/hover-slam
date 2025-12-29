from dataclasses import dataclass

import numpy as np


@dataclass
class Shape:
    nodes: np.ndarray
    edges: np.ndarray

    def __post_init__(self):
        # ensure that nodes is a 2d array where dimension 1 has size 2
        if self.nodes.ndim != 2:
            raise ValueError(f"nodes must be a 2D array, got {self.nodes.ndim}D")
        if self.nodes.shape[1] != 2:
            raise ValueError(f"nodes must have size 2 in dimension 1, got {self.nodes.shape[1]}")

        # ensure that nodes has integer type
        if not np.issubdtype(self.nodes.dtype, np.integer):
            raise ValueError(f"nodes must have integer type, got {self.nodes.dtype}")

        # ensure that edges is a 3d array where dimensions 1 and 2 both have size 2
        if self.edges.ndim != 3:
            raise ValueError(f"edges must be a 3D array, got {self.edges.ndim}D")
        if self.edges.shape[1] != 2:
            raise ValueError(f"edges must have size 2 in dimension 1, got {self.edges.shape[1]}")
        if self.edges.shape[2] != 2:
            raise ValueError(f"edges must have size 2 in dimension 2, got {self.edges.shape[2]}")

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

    def rotations(self):
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
        # Rotation matrices for hex grid (applied as coords @ matrix.T)
        rotation_matrices = [
            np.array([[1, 0], [0, 1]]),      # 0°
            np.array([[0, -1], [1, 1]]),     # 60°
            np.array([[-1, -1], [1, 0]]),    # 120°
            np.array([[-1, 0], [0, -1]]),    # 180°
            np.array([[0, 1], [-1, -1]]),    # 240°
            np.array([[1, 1], [-1, 0]]),     # 300°
        ]

        rotated_shapes = []

        for matrix in rotation_matrices:
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

            rotated_shapes.append(Shape(nodes=rotated_nodes, edges=rotated_edges))

        return rotated_shapes


DOODADS = [
    Shape(
        nodes=np.array(
            [
                [0, 0],
                [4, 0],
                [2, 1],
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
    )
]
