# SimProb: A simulation-based inference library optimized for readability

## Overview

SimProb is a minimalist, readable library for probabilistic simulation and state inference.
It generalizes Kalman filters, Hidden Markov Models, and other probabilistic tracking methods.

At its core, SimProb estimates the evolving state of a system over time from noisy and incomplete observations.

## Why SimProb?

Unlike other libraries designed for a specific filtering method, SimProb takes a generic perspective, resulting in a clean API and readable code.

For example, in Kalman filtering, most implementations
([pykalman](https://pypi.org/project/pykalman), [filterpy](https://pypi.org/project/filterpy), [zziz/kalman-filter](https://github.com/zziz/kalman-filter))
manage separate variables for the mean and covariance of multivariate normal distributions.

In contrast, SimProb's generic approach inherently requires treating state distributions as first-class objects.
To achieve this, SimProb encapsulates the mean and covariance in a `MultivariateNormal` class,
which happens to result in reduced boilerplate and clearer code.

### Trade-offs

Other libraries offer additional features, such as support for Unscented Kalman filters,
and are likely more performant and better tested for specific use cases.
