#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: test_model.py
PROJECT: rocket_landing
ORIGINAL AUTHOR: rcardenas
DATE CREATED: 13 May 2023

Unit tests for model.py.
"""

import pytest

import simulations.rocket_landing.model as model


class Test_d2x_dt2:
    """Tests that d2x_dt2() meets the following expectations:
    - Result is a float.
    - Result is zero given zero planet mass and zero thrust.
    - Result is positive given zero planet mass and positive thrust.
    - Result is negative given zero planet mass and negative thrust.
    - Result is negative given positive planet mass and zero thrust.
    - Raises ValueError if r_rocket is nonpositive.
    - Raises ValueError if mass_rocket is nonpositive.
    """
    @pytest.mark.parametrize("mass_planet", [7.3e22, 0.])
    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    @pytest.mark.parametrize("thrust", [1e3, 1., 0., -1., -1e3])
    def test_result_is_float(self, mass_planet, r_rocket, mass_rocket, thrust):
        result = model.d2x_dt2(
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            mass_rocket=mass_rocket,
            thrust=thrust
        )
        assert isinstance(result, float)

    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    def test_result_zero_given_zero_planet_mass_and_zero_thrust(self, r_rocket, mass_rocket):
        result = model.d2x_dt2(
            mass_planet=0.,
            r_rocket=r_rocket,
            mass_rocket=mass_rocket,
            thrust=0.
        )
        assert result == 0.

    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    @pytest.mark.parametrize("thrust", [9e9, 10., 1., 1e-9])
    def test_result_positive_given_zero_planet_mass_and_positive_thrust(self, r_rocket, mass_rocket, thrust):
        result = model.d2x_dt2(
            mass_planet=0.,
            r_rocket=r_rocket,
            mass_rocket=mass_rocket,
            thrust=thrust
        )
        assert result > 0.

    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    @pytest.mark.parametrize("thrust", [-9e9, -10., -1., -1e-9])
    def test_result_negative_given_zero_planet_mass_and_negative_thrust(self, r_rocket, mass_rocket, thrust):
        result = model.d2x_dt2(
            mass_planet=0.,
            r_rocket=r_rocket,
            mass_rocket=mass_rocket,
            thrust=thrust
        )
        assert result < 0.

    @pytest.mark.parametrize("mass_planet", [7.3e22, 1., 3e-9])
    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    def test_result_negative_given_positive_planet_mass_and_zero_thrust(self, mass_planet, r_rocket, mass_rocket):
        result = model.d2x_dt2(
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            mass_rocket=mass_rocket,
            thrust=0.
        )
        assert result < 0.

    @pytest.mark.parametrize("mass_planet", [7.3e22, 0.])
    @pytest.mark.parametrize("r_rocket", [-5.0e10, -1.8e6, -1., -3e-9, 0.])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1.])
    @pytest.mark.parametrize("thrust", [1e3, 1., 0., -1., -1e3])
    def test_raises_value_error_if_r_rocket_nonpositive(self, mass_planet, r_rocket, mass_rocket, thrust):
        with pytest.raises(AssertionError) as e:
            result = model.d2x_dt2(
                mass_planet=mass_planet,
                r_rocket=r_rocket,
                mass_rocket=mass_rocket,
                thrust=thrust
            )
        assert isinstance(e.value, AssertionError)

    @pytest.mark.parametrize("mass_planet", [7.3e22, 0.])
    @pytest.mark.parametrize("r_rocket", [5.0e10, 1.8e6])
    @pytest.mark.parametrize("mass_rocket", [-5e3, -1., -3e-9, 0.])
    @pytest.mark.parametrize("thrust", [1e3, 1., 0., -1., -1e3])
    def test_raises_value_error_if_mass_rocket_nonpositive(self, mass_planet, r_rocket, mass_rocket, thrust):
        with pytest.raises(AssertionError) as e:
            result = model.d2x_dt2(
                mass_planet=mass_planet,
                r_rocket=r_rocket,
                mass_rocket=mass_rocket,
                thrust=thrust
            )
        assert isinstance(e.value, AssertionError)