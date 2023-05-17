#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: test_solver.py
PROJECT: personal
ORIGINAL AUTHOR: rcardenas
DATE CREATED: 16 May 2023

Unit tests for solver.py.
"""

import pytest

import numpy as np

import simulations.rocket_landing.solver as solver


@pytest.mark.parametrize("r_planet, r0", [(1e6, 2.3e6), (3.2e8, 6.1e8), (5.0e7, 3e8)])
@pytest.mark.parametrize("dr", np.arange(1e3, 1e6, 1e5))
class Test_create_r_vector:
    """Tests that create_r_vector() meets the following expectations:
    - Result[0] equals r_planet.
    - Result[-1] is between r0 and (r0 + dr).
    - Elements are equally spaced by dr.
    """

    def test_result_first_element_equals_r_planet(self, r_planet, r0, dr):
        result = solver.create_r_vector(r_planet=r_planet, r0=r0, dr=dr)
        assert result[0] == r_planet

    def test_result_last_element_within_expected_bounds(self, r_planet, r0, dr):
        result = solver.create_r_vector(r_planet=r_planet, r0=r0, dr=dr)
        assert r0 <= result[-1] <= (r0 + dr)

    def test_elements_equally_spaced_by_dr(self, r_planet, r0, dr):
        result = solver.create_r_vector(r_planet=r_planet, r0=r0, dr=dr)
        assert np.allclose(np.diff(result), dr)


class Test_get_index:
    """Tests that get_index() meets the following expectations:
    - Returns exact index for values in x_vector.
    - Returns floor index for values between x_vector.
    - Returns max index for values above x_vector[-1].
    - Raises assertion error for values below x_vector[0].
    - Raises assertion error for x_vectors of shape (1,).
    """

    @pytest.mark.parametrize(
        "x_vector",
        [
            np.arange(10),
            np.arange(10.0),
            np.arange(-10, 10, 3),
            np.arange(-10.0, 5, 1.7),
        ],
    )
    def test_returns_exact_index_for_values_between_x_vector(self, x_vector):
        for x in x_vector:
            idx = solver.get_index(x_vector=x_vector, x=x)
            assert x_vector[idx] == x

    @pytest.mark.parametrize(
        "x_vector",
        [
            np.arange(10),
            np.arange(10.0),
            np.arange(-10, 10, 3),
            np.arange(-10.0, 5, 1.7),
        ],
    )
    @pytest.mark.parametrize("x", np.linspace(-10, 10, 43))
    def test_returns_floor_index_for_values_between_x_vector(self, x_vector, x):
        if (x not in x_vector) and (x_vector[0] < x < x_vector[-1]):
            idx = solver.get_index(x_vector=x_vector, x=x)
            assert x_vector[idx] < x < x_vector[idx + 1]

    @pytest.mark.parametrize(
        "x_vector",
        [
            np.arange(10),
            np.arange(10.0),
            np.arange(-10, 10, 3),
            np.arange(-10.0, 5, 1.7),
        ],
    )
    @pytest.mark.parametrize("x", np.linspace(20, 300, 43))
    def test_returns_max_index_for_values_above_x_vector_1(self, x_vector, x):
        result = solver.get_index(x_vector=x_vector, x=x)
        assert result == x_vector.shape[0]

    @pytest.mark.parametrize(
        "x_vector",
        [
            np.arange(10),
            np.arange(10.0),
            np.arange(-10, 10, 3),
            np.arange(-10.0, 5, 1.7),
        ],
    )
    @pytest.mark.parametrize("x", np.linspace(-20, -0.1, 43))
    def test_raises_assertion_error_for_values_below_x_vector_0(self, x_vector, x):
        if x < x_vector[0]:
            with pytest.raises(AssertionError) as e:
                solver.get_index(x_vector=x_vector, x=x)
            assert isinstance(e.value, AssertionError)

    @pytest.mark.parametrize(
        "x_vector",
        [
            np.array([0]),
            np.array([0.0]),
            np.array([1]),
            np.array([1.0]),
            np.array([-1]),
            np.array([-1.0]),
        ],
    )
    @pytest.mark.parametrize("x", np.linspace(10, 20, 23))
    def test_raises_assertion_error_for_x_vectors_of_shape_one(self, x_vector, x):
        with pytest.raises(AssertionError) as e:
            solver.get_index(x_vector=x_vector, x=x)
        assert isinstance(e.value, AssertionError)
