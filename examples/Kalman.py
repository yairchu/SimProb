import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pathlib
    import scipy
    import sys

    sys.path.append(pathlib.Path(__file__).parents[1].as_posix())
    return np, plt, scipy


@app.cell
def _(np, plt):
    accel_std = 0.03
    velocity = np.cumsum(accel_std * np.random.normal(size=80))
    position = np.cumsum(velocity)
    position -= np.mean(position)
    noise_std = 0.5
    observations = position + noise_std * np.random.normal(size=len(position))

    def plot_measures():
        plt.plot(position, label="Path")
        plt.scatter(
            np.arange(len(position)), observations, label="Noisy path observations"
        )

    plot_measures()
    plt.legend()
    plt.gca()
    return accel_std, noise_std, observations, plot_measures


@app.cell
def _(accel_std, noise_std, np, observations, plot_measures, plt, scipy):
    import simprob
    import simprob.kalman as kalman

    dims = 2

    kobs = [
        kalman.MultivariateNormal(np.array([x]), np.array([[noise_std**2]]))
        for x in observations
    ]
    steps = simprob.prepend_all(
        [
            kalman.KalmanTransition([[1.0, 1], [0, 1]]),
            kalman.add_process_noise([[0, 0], [0, accel_std**2]]),
        ],
        map(simprob.fuse, kobs[1:]),
    )

    kalman_comb = list(simprob.simulate(kobs[0], steps))[::3]

    plot_measures()

    means = np.array([x.mean[0] for x in kalman_comb])
    stds = np.array([x.covar[0, 0] ** 0.5 for x in kalman_comb])
    [means_plot] = plt.plot(means, label="Path estimation at each time with Kalman")
    for p in np.linspace(0.05, 1, 7)[:-1]:
        s = scipy.stats.chi2(1).isf(p) ** 0.5
        for mult in [s, -s]:
            plt.plot(means + mult * stds, color=means_plot.get_color(), alpha=p)
    plt.ylim(observations.min(), observations.max())
    plt.legend()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
