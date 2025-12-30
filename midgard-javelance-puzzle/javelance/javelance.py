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
    0: Shape.from_sets(
        nodes={
            (-1, 5),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
        }
    ),  # lower left
    1: Shape.from_sets(
        nodes={
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 1),
            (10, 2),
        }
    ),  # lower middle
    2: Shape.from_sets(
        nodes={
            (11, 1),
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (14, 2),
            (15, 0),
            (15, 1),
            (15, 2),
            (16, 0),
            (16, 1),
            (16, 2),
            (17, 0),
            (17, 1),
            (17, 2),
            (18, 0),
            (18, 1),
            (18, 2),
            (19, 1),
            (19, 2),
            (20, 1),
        }
    ),  # lower right
    3: Shape.from_sets(
        nodes={
            (13, 3),
            (13, 4),
            (13, 5),
            (14, 3),
            (14, 4),
            (14, 5),
            (15, 3),
            (15, 4),
            (15, 5),
            (16, 3),
            (16, 4),
            (17, 3),
            (18, 3),
            (19, 3),
        }
    ),  # right center
    4: Shape.from_sets(
        nodes={
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (13, 6),
            (13, 7),
            (13, 10),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (16, 6),
            (16, 7),
            (16, 8),
            (17, 5),
            (17, 6),
        }
    ),  # upper right
    5: Shape.from_sets(
        nodes={
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (11, 3),
            (11, 4),
        }
    ),  # upper center right
    6: Shape.from_sets(
        nodes={
            (1, 10),
            (2, 9),
            (2, 10),
            (3, 8),
            (3, 9),
            (4, 7),
            (5, 6),
            (5, 9),
            (5, 10),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
        }
    ),  # upper center left
    7: Shape.from_sets(
        nodes={
            (-3, 7),
            (-3, 8),
            (-3, 9),
            (-2, 6),
            (-2, 7),
            (-2, 8),
            (-1, 6),
            (-1, 7),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (2, 6),
            (2, 7),
            (2, 8),
            (3, 6),
            (3, 7),
            (4, 6),
        }
    ),  # upper left
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
