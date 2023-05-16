#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: cost_function.py
PROJECT: rocket_landing
ORIGINAL AUTHOR: rcardenas
DATE CREATED: 16 May 2023

Contains cost functions to be used in the dynamic programming solver for the rocket landing control problem.
"""

from typing import Optional

import numba
import numpy as np


@numba.njit
def soft_landing(
    r_planet: Optional[float],
    r_rocket: Optional[float],
    v_rocket: Optional[float],
    t: Optional[float],
    tf: Optional[float],
    eps: float = 0.0,
) -> float:
    """For terminal stage, assigns infinite cost to any altitude except ground level (r=r_surface). Penalizes hitting
    the ground at high speeds."""
    cost = 0.0
    if t >= tf:
        if -eps <= (r_rocket - r_planet) < eps:
            cost = v_rocket**2
        else:
            cost = np.inf
    return cost


@numba.njit
def quickest_time(
    r_planet: Optional[float],
    r_rocket: Optional[float],
    v_rocket: Optional[float],
    t: Optional[float],
    tf: Optional[float],
    eps: float = 0.0,
) -> float:
    """For terminal stage, assigns infinite cost to any altitude except ground level (r=r_surface). Penalizes hitting
    the ground at high speeds. Also penalizes total time in flight."""
    cost = t**2
    if t >= tf:
        if -eps <= (r_rocket - r_planet) < eps:
            cost = v_rocket**2 + t**2
        else:
            cost = np.inf
    return cost
