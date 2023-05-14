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
import numpy as np

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


class Test_integrate:
    """Tests that integrate() meets the following expectations:
        - Result is a numpy.ndarray.
        - Result has shape (2,).
        - Result[0] equals r_rocket when mass_planet, thrust, and v_rocket are all zero.
        - Result[1] equals v_rocket when mass_planet and thrust are both zero.
        - Result[0] decreases given zero mass_planet, zero thrust, and negative v_rocket.
        - Result[0] decreases given positive mass_planet, zero thrust, and zero v_rocket.
        - Result[0] increases given zero_mass_planet, zero thrust, and positive v_rocket.
        - Result[1] decreases given zero mass_planet and negative thrust.
        - Result[1] decreases given positive mass_planet and zero thrust.
        - Result[1] increases given zero mass_planet and positive thrust.
        """

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("mass_planet", [6e24, 2e-9, 0.])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("thrust", [9e9,1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_is_ndarray(self, v_rocket, mass_planet, r_rocket, thrust, mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            thrust=thrust,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("mass_planet", [6e24, 2e-9, 0.])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("thrust", [9e9, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_has_expected_shape(self, v_rocket, mass_planet, r_rocket, thrust, mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            thrust=thrust,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result.shape == (2,)

    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [5., 1., 1e-3])
    def test_first_element_is_r_rocket_when_mass_planet_and_thrust_and_v_rocket_all_zero(self, r_rocket, mass_rocket, dt):
        result = model.integrate(
            v_rocket=0.,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert np.isclose(result[0], r_rocket)

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [5., 1., 1e-3])
    def test_second_element_is_v_rocket_when_mass_planet_and_thrust_all_zero(self, v_rocket, r_rocket, mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert np.isclose(result[1], v_rocket)

    @pytest.mark.parametrize("v_rocket", [-4e3, -1e-2, -2., -5e4])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_0_decreases_given_zero_mass_planet_zero_thrust_and_negative_v_rocket(self, v_rocket, r_rocket, mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[0] < r_rocket

    @pytest.mark.parametrize("mass_planet", [6e24])
    @pytest.mark.parametrize("r_rocket", [6.2e6, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_0_decreases_given_positive_mass_planet_zero_thrust_and_zero_v_rocket(self, mass_planet, r_rocket, mass_rocket, dt):
        result = model.integrate(
            v_rocket=0.,
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[0] < r_rocket

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-2])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_0_increases_given_zero_mass_planet_zero_thrust_and_positive_v_rocket(self, v_rocket, r_rocket,
                                                                                         mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[0] > r_rocket

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-9])
    @pytest.mark.parametrize("thrust", [-9e9, -1e-3])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_1_decreases_given_zero_mass_planet_and_negative_thrust(self, v_rocket, r_rocket, thrust,
                                                                           mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=thrust,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[1] < v_rocket

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("mass_planet", [6e24, 2e14])
    @pytest.mark.parametrize("r_rocket", [6.2e6, 1e-9])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-1])
    def test_result_1_decreases_given_positive_mass_planet_and_zero_thrust(self, v_rocket, mass_planet, r_rocket,
                                                                           mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=mass_planet,
            r_rocket=r_rocket,
            thrust=0.,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[1] < v_rocket

    @pytest.mark.parametrize("v_rocket", [4e3, 1e-9, 0., -2., -5e4])
    @pytest.mark.parametrize("r_rocket", [6.2e10, 1e-3])
    @pytest.mark.parametrize("thrust", [9e9, 1e-3])
    @pytest.mark.parametrize("mass_rocket", [5e3, 1e-9])
    @pytest.mark.parametrize("dt", [1., 1e-3])
    def test_result_1_increases_given_zero_mass_planet_and_positive_thrust(self, v_rocket, r_rocket, thrust, mass_rocket, dt):
        result = model.integrate(
            v_rocket=v_rocket,
            mass_planet=0.,
            r_rocket=r_rocket,
            thrust=thrust,
            mass_rocket=mass_rocket,
            dt=dt,
        )
        assert result[1] > v_rocket