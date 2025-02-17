import numpy as np
import random

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


def is_valid_position(x, y):
    "Check if the position is inside the board."
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def random_position():
    return tuple(random.randint(0, BOARD_SIZE - 1) for _ in "XY")


KNIGHT_MOVES = [
    (x, y) for ax, ay in [(2, 1), (1, 2)] for x in [-ax, ax] for y in [-ay, ay]
]


def get_possible_moves(position):
    "Return valid knight moves from a given position."
    x, y = position
    return [
        (x + dx, y + dy) for dx, dy in KNIGHT_MOVES if is_valid_position(x + dx, y + dy)
    ]


def random_knight_path(pos=None):
    "Simulates a random knight's tour on the chessboard."
    if pos is None:
        pos = random_position()
    yield pos
    while True:
        pos = random.choice(get_possible_moves(pos))
        yield pos


knight_moves_kernel = np.zeros([5, 5], dtype=bool)
for x, y in KNIGHT_MOVES:
    knight_moves_kernel[2 + x, 2 + y] = True
