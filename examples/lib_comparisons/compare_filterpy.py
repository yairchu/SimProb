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

    import filterpy.common
    import filterpy.kalman
    import simprob.kalman

    return filterpy, mo, np, plt, simprob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Multivariate Kalman Filters

    Comparing to examples from:

    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

    ### Kalman Filter predictions without observations

    (cells 14-17)
    """
    )
    return


@app.cell
def _(filterpy, np, simprob):
    start_x = np.array([10.0, 4.5])
    start_P = np.diag([500, 49])
    print("filterpy:")
    x = start_x
    _P = start_P
    F = np.array([[1, 0.1], [0, 1]])
    for _ in range(5):
        x, _P = filterpy.kalman.predict(x=x, P=_P, F=F, Q=0)
        print("x =", x)
    print(_P)
    print()
    print("simprob:")
    for r in simprob.simulate(
        simprob.kalman.MultivariateNormal(start_x, start_P),
        [simprob.kalman.KalmanTransition(F)] * 5,
    ):
        print("x =", r.mean)
    print(r.covar)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Complete Kalman Filter with observations

    (cells 4 & 27-31)
    """
    )
    return


@app.cell
def _(filterpy, np, plt, simprob):
    def compute_dog_data(z_var, process_var, count=1, dt=1.0):
        """returns track, measurements 1D ndarrays"""
        x, vel = (0.0, 1.0)
        z_std = np.sqrt(z_var)
        p_std = np.sqrt(process_var)
        xs, zs = ([], [])
        for _ in range(count):
            v = vel + np.random.randn() * p_std
            x += v * dt
            xs.append(x)
            zs.append(x + np.random.randn() * z_std)
        return (np.array(xs), np.array(zs))

    def pos_vel_filter(x, P, R, Q=0.0, dt=1.0):
        """Returns a KalmanFilter which implements a
        constant velocity model for a state [x dx].T
        """
        kf = filterpy.kalman.KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([x[0], x[1]])
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0]])
        kf.R *= R
        if np.isscalar(P):
            kf.P *= P
        else:
            kf.P[:] = P
        if np.isscalar(Q):
            kf.Q = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        else:
            kf.Q[:] = Q
        return kf

    plt.title("filterpy and simprob produce same results")
    dt = 0.1
    count = 50
    R = 10
    Q = 0.01
    track, zs = compute_dog_data(R, Q, count)
    plt.plot(track, label="True state")
    plt.scatter(np.arange(len(zs)), zs, label="Observations")
    x0 = (0.0, 0.0)
    _P = np.diag([500.0, 49.0])
    kf = pos_vel_filter(x0, R=R, P=_P, Q=Q, dt=dt)
    init_Q = kf.Q
    xs, cov = ([], [])
    for z in zs:
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)
    xs, cov = (np.array(xs), np.array(cov))
    plt.plot(xs[:, 0], label="Kalman (filterpy)")
    res = list(
        simprob.simulate(
            simprob.kalman.MultivariateNormal(np.array(x0), _P),
            simprob.prepend_all(
                [
                    simprob.kalman.KalmanTransition(kf.F),
                    simprob.kalman.add_process_noise(init_Q),
                ],
                [
                    simprob.fuse(simprob.kalman.MultivariateNormal(o[None], kf.R))
                    for o in zs
                ],
            ),
        )
    )[::3][1:]
    plt.plot([r.mean[0] for r in res], "--", label="Kalman (simprob)")
    plt.legend()
    (abs(xs - [r.mean for r in res]) < 1e-10).all()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
