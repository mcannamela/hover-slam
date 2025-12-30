import functools
import operator

import numpy as np

from javelance.shapes import Shape

JAVELANCE_COLOR = "Maroon"
JAVELANCE_FORBIDDEN_COLOR = "LightGrey"
JAVELANCE_VBOX = Shape.vertical_box(width=21, height=11, mean_color=JAVELANCE_COLOR)
JAVELANCE_GRID_SHAPE = Shape.vertical_box(
    width=23, height=13, mean_color=JAVELANCE_COLOR
).translate(np.array([-1, -1]))
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

JAVELANCE_REGIONS = {
    0: Shape.from_sets(),  # lower left
    1: Shape.from_sets(),  # lower middle
    2: Shape.from_sets(),  # lower right
    3: Shape.from_sets(),  # right center
    4: Shape.from_sets(),  # upper right
    5: Shape.from_sets(),  # upper center right
    6: Shape.from_sets(),  # upper center left
    7: Shape.from_sets(),  # upper left
}

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
