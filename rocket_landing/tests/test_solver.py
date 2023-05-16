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
