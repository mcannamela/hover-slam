from dataclasses import dataclass

import numpy as np


@dataclass
class Shape:
    nodes: np.ndarray
    edges: np.ndarray

    def __post_init__(self):
        # ensure that nodes is a 2d array where dimension 1 has size 2
        # ensure that edges is a 3d array where dimensions 1 and 2 both have size 2
        # raise if any nodes or edges are duplicated
        # verify that all edges are valid i.e. the hexes they address are adjacent
        pass


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
