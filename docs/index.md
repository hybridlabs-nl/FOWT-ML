#

{% include-markdown "../README.md" end="# FOWT-ML" %}

## FOWT-ML: Floating Offshore Wind Turbine Machine Learning Kit

FOWT-ML is a generic machine learning toolkit developed for Hyrbid testing of
Floating Offshore Wind Turbines (FOWTs). It provides a set of tools and
algorithms to facilitate reaserch on machine learning techniques (specifically
multi-output regressions) for real-time hybrid testing setups in wind tunnels
or wave basins.

The package is designed to be flexible and extensible, allowing users to
customize and adapt it to their specific needs and requirements. It includes
various machine learning algorithms from linear regression to neural networks,
as well as tools for data preprocessing, model evaluation and comparison, and
model publication.

## Multi-output regression in hyrbid testing

In real-time hybrid testing of FOWTs, it is often necessary to use numerical
models to simulate missing components or dynamics that cannot be physically
measured. For example, in wind tunnels, the hydrodynamic forces acting on the
floating platform may not be directly measurable, and thus a numerical model is
used to estimate these forces based on the measured wind loads and platform
motions. On the other hand, in wave basins, the aerodynamic forces on the wind
turbine may not be directly measurable, and a numerical model is used to
estimate these forces based on the measured wave loads and turbine motions.

Machine learning techniques can be employed to predict the missing forces in one
lab based on the available measurements from the other lab. This is where
multi-output regression comes into play, as it allows for the simultaneous
prediction of multiple outputs i.e., the missing forces in 6 degrees of freedom
(DOF) of the floating platform.
