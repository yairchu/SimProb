import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Hidden Markov Model Knight's Tour example

    A Knight went for a Tour on the chess board.

    A drunk reporter wrote down the knight's positions, and we trust each report as having a 50% of being random.
    Can we infer where the knight has been?
    """
    )
    return


@app.cell
def _():
    import itertools
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pathlib
    import random
    import sys

    sys.path.append(pathlib.Path(__file__).parents[1].as_posix())
    return itertools, mo, np, plt, random


@app.cell
def _(itertools, plt, random):
    import knight_tour

    path = list(itertools.islice(knight_tour.random_knight_path(), 12))

    reported_path = [
        random.choice([pos, knight_tour.random_position()]) for pos in path
    ]

    plt.axis("equal")
    plt.plot(*zip(*path), ".-", label="Actual knight tour")
    plt.plot(*zip(*reported_path), ".-", label="Reported knight tour", alpha=0.5)
    plt.legend()
    plt.gca()
    return knight_tour, path, reported_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Inference using SimProb""")
    return


@app.cell
def _(knight_tour, np, path, plt, reported_path):
    import simprob
    import simprob.smoothing as smoothing
    import simprob.hidden_markov as hmm

    def reported_path_probabilites(pos):
        """
        Position reported by drunk reporter has 50% chance of being random,
        and 50% of being accurate.

        This function computes the resulting probability distribution.
        """
        res = np.ones(knight_tour.BOARD_SHAPE)
        res[pos[::-1]] += res.sum()
        return hmm.Histogram(res)

    inferred = np.asarray(
        list(
            smoothing.forward_backward(
                reported_path_probabilites(reported_path[0]),
                simprob.prepend_all(
                    [hmm.ConvolutionTransition(knight_tour.knight_moves_kernel)],
                    [
                        simprob.fuse(reported_path_probabilites(o))
                        for o in reported_path[1:]
                    ],
                ),
                hmm.Histogram.empty(knight_tour.BOARD_SHAPE),
            )
        )
    )[::2]

    print("Inference of knight's possible positions from observations")
    for h, (real_x, real_y), (rep_x, rep_y) in zip(inferred, path, reported_path):
        plt.figure(figsize=(2, 2))
        plt.imshow(h.probs)
        plt.colorbar()
        plt.scatter([real_x], [real_y], color="white")
        plt.scatter([rep_x], [rep_y], marker="x", color="red")
        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
