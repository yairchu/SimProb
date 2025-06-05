import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Knight's Tour example

    Example problems with a Knight's Tour on the chess board.
    """
    )
    return


@app.cell
def _():
    import itertools
    import marimo as mo
    import matplotlib.pyplot as plt
    import pathlib
    import numpy as np
    import os
    import random
    import scipy
    import sys

    sys.path.append(pathlib.Path(__file__).parents[1].as_posix())
    return itertools, mo, np, plt, random, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Number of options for tour between oppsite corners

    In 8 steps, how many paths can a knight take between opposite corners?
    """
    )
    return


@app.cell
def _(np, plt, scipy):
    import knight_tour
    import simprob


    def advance_knight_counts(counts):
        return scipy.signal.convolve(
            counts, knight_tour.knight_moves_kernel, mode="same"
        )


    start = np.zeros(knight_tour.BOARD_SHAPE, dtype=int)
    start[0, 0] = 1
    n_steps = 8
    for state in simprob.simulate(start, [advance_knight_counts] * n_steps):
        plt.figure(figsize=(2, 2))
        plt.imshow(state)
        plt.colorbar()
        plt.show()
    print(
        f"Number of paths of {n_steps} steps between opposite corners: {state[-1, -1]}"
    )
    return knight_tour, simprob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Infer path from partial observations.

    Given sporadic observations of Knight row, column, diagonal or quadrant at specific times,
    could we infer where the knight have been?

    ### Randomize path
    """
    )
    return


@app.cell
def _(itertools, knight_tour, plt):
    path = list(itertools.islice(knight_tour.random_knight_path(), 15))
    plt.axis("equal")
    plt.title("The knight's tour")
    plt.plot(*zip(*path), ".-")
    plt.gca()
    return (path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Randomize observations""")
    return


@app.cell
def _(knight_tour, np, path, plt, random):
    observations = []
    for x, y in path:
        kind = random.choice(["row", "col", "diag0", "diag1", "quadrant"])
        obs = np.zeros(knight_tour.BOARD_SHAPE, dtype=bool)
        if kind == "row":
            obs[:, y] = True
        elif kind == "col":
            obs[x] = True
        elif kind == "diag0":
            for _i in range(-min(x, y), 8 - max(x, y)):
                obs[x + _i, y + _i] = True
        elif kind == "diag1":
            for _i in range(-min(7 - x, y), 8 - max(7 - x, y)):
                obs[x - _i, y + _i] = True
        else:
            obs[
                slice(4) if x < 4 else slice(4, None),
                slice(4) if y < 4 else slice(4, None),
            ] = True
        observations.append(obs)
    observations = np.asarray(observations)
    print("Partial observations of knight's position over time")
    _, _ax = plt.subplots(ncols=len(observations))
    for _a, o, (x, y) in zip(_ax, observations, path):
        o = o.astype(int)
        o[x, y] += 1
        _a.imshow(o)
        _a.set_xticks([])
        _a.set_yticks([])
    plt.gca()
    return (observations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Infer path using simprob""")
    return


@app.cell
def _(knight_tour, np, observations, plt, scipy, simprob):
    import simprob.smoothing as smoothing


    class KnightTransition:
        def __call__(self, mask: np.ndarray) -> np.ndarray:
            return (
                scipy.signal.convolve(
                    mask.astype(int), knight_tour.knight_moves_kernel, mode="same"
                )
                > 0
            )

        @property
        def inv(self):
            return self


    inferred = np.asarray(
        list(
            smoothing.forward_backward(
                observations[0],
                sum(
                    (
                        [KnightTransition(), simprob.fuse(o)]
                        for o in observations[1:]
                    ),
                    [],
                ),
                np.ones(knight_tour.BOARD_SHAPE, dtype=bool),
            )
        )[::2]
    )
    print("Inference of knight's possible positions from observations")
    _, _ax = plt.subplots(ncols=len(inferred))
    for _a, _i in zip(_ax, inferred + observations.astype(int)):
        _a.imshow(_i)
        _a.set_xticks([])
        _a.set_yticks([])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Infering the path with probabilities

    See `Hidden Markov Model.py` for an example
    """
    )
    return


if __name__ == "__main__":
    app.run()
