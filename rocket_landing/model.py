#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: model.py
PROJECT: rocket_landing
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 13 May 2023

Simulates a simple rocket ship landing on a celestial body in the presence of a gravitational field.
"""

import matplotlib.pyplot as plt
import numba
import numpy as np


@numba.njit
def d2x_dt2(
    mass_planet: float, r_rocket: float, mass_rocket: float, thrust: float
) -> float:
    """Rocket acceleration due to gravity and thrust."""
    assert r_rocket > 0.
    assert mass_rocket > 0.
    grav_const = 6.6743e-11
    return -grav_const * mass_planet / r_rocket**2 + thrust / mass_rocket


@numba.njit
def integrate(
    v_rocket: float,
    mass_planet: float,
    r_rocket: float,
    thrust: float,
    mass_rocket: float,
    dt: float,
) -> np.ndarray:
    """Integrates model forward one full step."""
    a_rocket = d2x_dt2(
        mass_planet=mass_planet,
        r_rocket=r_rocket,
        mass_rocket=mass_rocket,
        thrust=thrust,
    )
    r_rocket_new = 0.5 * a_rocket * dt**2 + v_rocket * dt + r_rocket
    v_rocket_new = a_rocket * dt + v_rocket
    return np.array([r_rocket_new, v_rocket_new])


@numba.njit
def simulate(
    tf: float,
    v0: float,
    r0: float,
    m_planet: float,
    r_planet: float,
    dt: float,
    m_rocket: float,
    thrust: float,
) -> np.ndarray:
    """Simulates a rocket falling toward a planet."""
    n_steps = int(tf // dt)
    states = np.full((n_steps, 3), fill_value=np.nan, dtype=np.float64)
    states[0, 0] = 0.0
    states[0, 1] = r0
    states[0, 2] = v0
    for i in range(1, n_steps):
        if states[i - 1, 1] >= r_planet:
            new_state = integrate(
                r_rocket=states[i - 1, 1],
                v_rocket=states[i - 1, 2],
                mass_planet=m_planet,
                dt=dt,
                mass_rocket=m_rocket,
                thrust=thrust,
            )
            states[i, 0] = i * dt
            states[i, 1] = new_state[0]
            states[i, 2] = new_state[1]

    return states


if __name__ == "__main__":
    v0 = 30
    mass_planet = 7.34e22
    mass_rocket = 5000
    thrust = 7500.0
    radius_planet = 1.74e6
    position_rocket = 1.75e6
    dt = 1
    tf = 1000.0
    states = simulate(
        tf=tf,
        r0=position_rocket,
        v0=v0,
        m_planet=mass_planet,
        r_planet=radius_planet,
        dt=dt,
        m_rocket=mass_rocket,
        thrust=thrust,
    )
    t = states[:, 0]
    r = states[:, 1]
    v = states[:, 2]

    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(t, r)
    ax[0].set_xlim(0, tf)
    ax[0].grid()
    ax[1].plot(t, v)
    ax[1].set_xlim(0, tf)
    ax[1].grid()
    plt.show()
