import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.python.orbit.orbital_elements import (
    OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv,
    OrbitalConstants, mean_anomaly_to_eccentric,
    eccentric_to_true_anomaly, true_to_eccentric_anomaly
)
from src.python.orbit.perturbations import J2Perturbation, AtmosphericDrag
from src.python.orbit.orbit_propagator import OrbitPropagator
from src.python.orbit.ekf_filter import ExtendedKalmanFilter


class TestOrbitalElements:
    def test_creation(self):
        elem = OrbitalElements(
            semi_major_axis=7000000.0,
            eccentricity=0.001,
            inclination=np.radians(97.5),
            raan=0.0,
            arg_of_perigee=0.0,
            true_anomaly=0.0
        )
        assert elem.semi_major_axis == 7000000.0
        assert elem.eccentricity == 0.001

    def test_to_array(self):
        elem = OrbitalElements(
            semi_major_axis=7000000.0,
            eccentricity=0.001,
            inclination=np.radians(97.5),
            raan=0.0,
            arg_of_perigee=0.0,
            true_anomaly=0.0
        )
        arr = elem.to_array()
        assert len(arr) == 6
        assert arr[0] == 7000000.0

    def test_rv_conversion_circular(self):
        r = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        v = np.array([0, np.sqrt(OrbitalConstants.EARTH_MU / r[0]), 0])

        elements = rv_to_orbital_elements(r, v)
        assert abs(elements.eccentricity) < 0.01

        r_back, v_back = orbital_elements_to_rv(elements)
        assert np.allclose(r, r_back, rtol=1e-3)
        assert np.allclose(v, v_back, rtol=1e-3)

    def test_rv_conversion_elliptical(self):
        elem = OrbitalElements(
            semi_major_axis=26600000.0,
            eccentricity=0.01,
            inclination=np.radians(55),
            raan=np.radians(45),
            arg_of_perigee=np.radians(90),
            true_anomaly=np.radians(30)
        )

        r, v = orbital_elements_to_rv(elem)
        elem_back = rv_to_orbital_elements(r, v)

        assert abs(elem.semi_major_axis - elem_back.semi_major_axis) / elem.semi_major_axis < 1e-6
        assert abs(elem.eccentricity - elem_back.eccentricity) < 1e-6
        assert abs(elem.inclination - elem_back.inclination) < 1e-6

    def test_anomaly_conversions(self):
        M = np.radians(45)
        e = 0.1

        E = mean_anomaly_to_eccentric(M, e)
        nu = eccentric_to_true_anomaly(E, e)
        E_back = true_to_eccentric_anomaly(nu, e)

        assert abs(E - E_back) < 1e-10


class TestPerturbations:
    def test_j2_initialization(self):
        j2 = J2Perturbation()
        assert j2 is not None

    def test_j2_acceleration(self):
        j2 = J2Perturbation()
        r = np.array([7000000.0, 0, 0])
        acc = j2.acceleration(r)

        assert len(acc) == 3
        assert not np.allclose(acc, np.zeros(3), atol=1e-10)
        assert abs(acc[2]) < abs(acc[0])

    def test_drag_initialization(self):
        drag = AtmosphericDrag()
        assert drag is not None

    def test_drag_density(self):
        drag = AtmosphericDrag()
        r = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        rho = drag.density(r)

        assert rho > 0
        assert rho < 1e-10

    def test_drag_acceleration(self):
        drag = AtmosphericDrag()
        r = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        v = np.array([0, 7700, 0])
        acc = drag.acceleration(r, v)

        assert len(acc) == 3
        assert acc[1] < 0


class TestOrbitPropagator:
    def test_initialization(self):
        prop = OrbitPropagator()
        assert prop is not None

    def test_two_body_propagation(self):
        prop = OrbitPropagator(use_j2=False, use_drag=False)

        r0 = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        v0 = np.array([0, np.sqrt(OrbitalConstants.EARTH_MU / r0[0]), 0])

        result = prop.propagate(r0, v0, t_span=(0, 600), dt=60)

        assert len(result.time) > 0
        assert result.position.shape[1] == 3
        assert result.velocity.shape[1] == 3

        r_end = result.position[-1]
        v_end = result.velocity[-1]

        energy_start = 0.5 * np.linalg.norm(v0)**2 - OrbitalConstants.EARTH_MU / np.linalg.norm(r0)
        energy_end = 0.5 * np.linalg.norm(v_end)**2 - OrbitalConstants.EARTH_MU / np.linalg.norm(r_end)

        assert abs(energy_start - energy_end) / abs(energy_start) < 1e-6

    def test_propagation_with_j2(self):
        prop = OrbitPropagator(use_j2=True, use_drag=False)

        elem = OrbitalElements(
            semi_major_axis=OrbitalConstants.EARTH_RADIUS + 400000,
            eccentricity=0.001,
            inclination=np.radians(97.5),
            raan=0.0,
            arg_of_perigee=0.0,
            true_anomaly=0.0
        )

        result = prop.propagate_elements(elem, t_span=(0, 600), dt=60)

        assert len(result.time) > 0
        assert len(result.elements) > 0

        if result.elements[0] and result.elements[-1]:
            raan_change = abs(result.elements[-1].raan - result.elements[0].raan)
            assert raan_change > 0

    def test_step_function(self):
        prop = OrbitPropagator()

        r0 = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        v0 = np.array([0, 7700, 0])

        r1, v1 = prop.step(r0, v0, 60.0)

        assert len(r1) == 3
        assert len(v1) == 3
        assert not np.allclose(r0, r1)

    def test_state_transition_matrix(self):
        prop = OrbitPropagator(use_j2=False, use_drag=False)

        r0 = np.array([OrbitalConstants.EARTH_RADIUS + 400000, 0, 0])
        v0 = np.array([0, 7700, 0])

        stm = prop.get_state_transition_matrix(r0, v0, 60.0)

        assert stm.shape == (6, 6)
        det = np.linalg.det(stm)
        assert abs(det - 1.0) < 0.1


class TestExtendedKalmanFilter:
    def test_initialization(self):
        prop = OrbitPropagator()
        ekf = ExtendedKalmanFilter(prop)
        assert ekf is not None
        assert ekf.state_dim == 6

    def test_initialize_from_rv(self):
        prop = OrbitPropagator()
        ekf = ExtendedKalmanFilter(prop)

        r = np.array([7000000.0, 0, 0])
        v = np.array([0, 7500.0, 0])
        ekf.initialize_from_rv(r, v, timestamp=1000.0)

        assert ekf.state.timestamp == 1000.0
        assert np.allclose(ekf.state.position, r)
        assert np.allclose(ekf.state.velocity, v)

    def test_predict(self):
        prop = OrbitPropagator(use_j2=False, use_drag=False)
        ekf = ExtendedKalmanFilter(prop)

        r = np.array([7000000.0, 0, 0])
        v = np.array([0, 7500.0, 0])
        ekf.initialize_from_rv(r, v, timestamp=0.0)

        state = ekf.predict(60.0, timestamp=60.0)

        assert state.timestamp == 60.0
        assert not np.allclose(state.position, r)
        assert state.P.shape == (6, 6)

    def test_update(self):
        prop = OrbitPropagator(use_j2=False, use_drag=False)
        ekf = ExtendedKalmanFilter(prop)

        r = np.array([7000000.0, 0, 0])
        v = np.array([0, 7500.0, 0])
        ekf.initialize_from_rv(r, v, timestamp=0.0)

        measurement = r + np.array([100.0, 50.0, 20.0])
        noise = np.eye(3) * 10.0**2

        state, innovation = ekf.update(measurement, noise, timestamp=0.0)

        assert len(innovation) == 3
        assert state.P[0, 0] < ekf.state.P[0, 0] + 1e-10

    def test_get_orbital_elements(self):
        prop = OrbitPropagator()
        ekf = ExtendedKalmanFilter(prop)

        r = np.array([7000000.0, 0, 0])
        v = np.array([0, 7500.0, 0])
        ekf.initialize_from_rv(r, v, timestamp=0.0)

        elements = ekf.get_orbital_elements()
        assert elements is not None
        assert elements.semi_major_axis > 0

    def test_process_gps_pseudorange(self):
        prop = OrbitPropagator()
        ekf = ExtendedKalmanFilter(prop)

        r = np.array([7000000.0, 0, 0])
        v = np.array([0, 7500.0, 0])
        ekf.initialize_from_rv(r, v, timestamp=0.0)

        pseudoranges = [
            {'prn': 1, 'pseudorange': 26500000.0 + 7000000.0},
            {'prn': 2, 'pseudorange': 26500000.0 + 7000000.0},
            {'prn': 3, 'pseudorange': 26500000.0 + 7000000.0},
        ]

        sat_positions = {
            1: np.array([26559700.0, 0, 0]),
            2: np.array([0, 26559700.0, 0]),
            3: np.array([0, 0, 26559700.0]),
        }

        state = ekf.process_gps_pseudorange(pseudoranges, sat_positions, 1.0)
        assert state is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
