import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import os
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pathlib
    import sys

    sys.path.append(pathlib.Path(__file__).parents[2].as_posix())

    import pykalman
    import simprob.kalman

    return mo, np, plt, pykalman, simprob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Kalman Filter example

    Part of example from https://github.com/pykalman/pykalman/blob/main/examples/standard/plot_filter.py

    For clarity, only one dimension is plotted
    """
    )
    return


@app.cell
def _(np, plt, pykalman, simprob):
    random_state = np.random.RandomState(0)
    transition_matrix = [[1, 0.1], [0, 1]]
    transition_offset = [-0.1, 0.1]
    observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
    observation_offset = [1.0, -1.0]
    transition_covariance = np.eye(2)
    observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
    initial_state_mean = [5, -5]
    initial_state_covariance = [[1, 0.1], [-0.1, 1]]

    # sample from model
    kf = pykalman.KalmanFilter(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        observation_covariance,
        transition_offset,
        observation_offset,
        initial_state_mean,
        initial_state_covariance,
        random_state=random_state,
    )
    states, observations = kf.sample(n_timesteps=50, initial_state=initial_state_mean)

    # estimate state with filtering and smoothing
    filtered_state_estimates = kf.filter(observations)[0]

    # draw estimates
    plt.plot(states[:, 0], label="True state")
    plt.scatter(np.arange(len(observations)), observations[:, 0], label="Observations")
    plt.plot(filtered_state_estimates[:, 0], label="Kalman (pykalman)")

    obs = [
        np.linalg.inv(np.array(observation_matrix))
        @ simprob.kalman.MultivariateNormal(mean=o, covar=observation_covariance)
        - np.array(observation_offset)
        for o in observations
    ]
    iters = simprob.prepend_all(
        [
            simprob.kalman.KalmanTransition(transition_matrix),
            simprob.kalman.add_process_noise(
                transition_covariance, -np.array(transition_offset)
            ),
        ],
        map(simprob.fuse, obs[1:]),
    )

    init = obs[0] & simprob.kalman.MultivariateNormal(
        mean=np.array(initial_state_mean), covar=np.array(initial_state_covariance)
    )
    res = list(simprob.simulate(init, iters))[::3]
    plt.plot([r.mean[0] for r in res], "--", label="Kalman (simprob)")
    plt.legend(loc="lower right")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
