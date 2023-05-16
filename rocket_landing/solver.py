#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: solver.py
PROJECT: rocket_landing
ORIGINAL AUTHOR: rcardenas
DATE CREATED: 16 May 2023

Dynamic programming code for optimizing thrust control during a rocket landing.
"""

from typing import Union

import numba
import numpy as np

import simulations.rocket_landing.cost_function as cf
from simulations.rocket_landing.model import d2x_dt2, integrate


@numba.njit
def create_v_vector(mass_rocket: float, mass_planet: float, r0: float, r_planet: float):
    a_rocket = d2x_dt2(
        mass_planet=mass_planet,
        r_rocket=r0,
        mass_rocket=mass_rocket,
        thrust=0.0,
    )
    """Creates the velocity dimension of our state space. We probably don't need more kinetic energy than approximately
     2x initial mechanical potential energy, which gives us an upper limit on speed."""
    approx_potential_energy = abs(2 * mass_rocket * a_rocket * (r0 - r_planet))
    max_speed = np.sqrt(2 * approx_potential_energy / mass_rocket).round()
    return np.arange(-max_speed, max_speed, 1)


@numba.njit
def create_r_vector(r_planet: float, r0: float, dr: float = 10.0) -> np.ndarray:
    """Creates the position dimension of our state space."""
    return np.arange(r_planet, r0 + dr, dr)


@numba.njit
def get_index(x_vector: np.ndarray, x: Union[float, int]):
    """Retrieves the index of an element in an evenly spaced, sorted ascending vector."""
    assert x >= x_vector[0]
    assert x_vector.shape[0] > 1
    dx = x_vector[1] - x_vector[0]
    idx = int((x - x_vector[0]) // dx)
    return min(x_vector.shape[0], idx)


@numba.njit
def populate_state_map(
    r_prediction_map: np.ndarray,
    v_prediction_map: np.ndarray,
    r_predicted_indices: np.ndarray,
    v_predicted_indices: np.ndarray,
    r_vector: np.ndarray,
    v_vector: np.ndarray,
    thrust_vector: float,
    mass_planet: float,
    mass_rocket: float,
    dt: float,
):
    """Computes the map from present states/actions to future states. Because this map is time-invariant, it can be
    cached and used as a lookup table to improve compute time."""
    for i in range(r_vector.shape[0]):
        for j in range(v_vector.shape[0]):
            for k in range(thrust_vector.shape[0]):
                new_state = integrate(
                    r_rocket=r_vector[i],
                    v_rocket=v_vector[j],
                    thrust=thrust_vector[k],
                    mass_planet=mass_planet,
                    dt=dt,
                    mass_rocket=mass_rocket,
                )
                r_new = new_state[0]
                v_new = new_state[1]
                r_prediction_map[k, i, j] = r_new
                v_prediction_map[k, i, j] = v_new
                r_predicted_indices[k, i, j] = get_index(x_vector=r_vector, x=r_new)
                v_predicted_indices[k, i, j] = get_index(x_vector=v_vector, x=v_new)


# @numba.njit
# def solve_backward(
#     tf: float,
#     dt: float,
#     r0: float,
#     v0: float,
#     r_planet: float,
#     mass_planet: float,
#     mass_rocket: float,
#     thrust_vector: np.ndarray,
#     dr: float=10.0
# ):
#     r_vector = create_r_vector(r_planet=r_planet, r0=r0, dr=dr)
#     v_vector = create_v_vector(mass_rocket=mass_rocket, mass_planet=mass_planet, r0=r0, r_planet=r_planet)
#
#     k_stages = int(tf // dt)
#     cost_matrix = np.full((k_stages, r_vector.shape[0], v_vector.shape[0]), fill_value=np.nan, dtype=np.float64)
#     control_matrix = np.full((k_stages, r_vector.shape[0], v_vector.shape[0]), fill_value=np.nan, dtype=np.float64)
#
#     r_prediction_map = np.full(
#         (thrust_vector.shape[0], r_vector.shape[0], v_vector.shape[0]),
#         fill_value=np.nan,
#         dtype=np.float64
#     )
#     r_predicted_indices = np.full(
#         (thrust_vector.shape[0], r_vector.shape[0], v_vector.shape[0]),
#         fill_value=np.nan,
#         dtype=np.int32
#     )
#     v_prediction_map = np.full(
#         (thrust_vector.shape[0], r_vector.shape[0], v_vector.shape[0]),
#         fill_value=np.nan,
#         dtype=np.float64
#     )
#     v_predicted_indices = np.full(
#         (thrust_vector.shape[0], r_vector.shape[0], v_vector.shape[0]),
#         fill_value=np.nan,
#         dtype=np.int32
#     )
