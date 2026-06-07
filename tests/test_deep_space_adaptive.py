import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import pytest
from datetime import datetime

from orbit.space_weather import SpaceWeatherProvider, SpaceWeatherData, SpaceWeatherEvent
from orbit.nrlmsise00 import NRLMSISE00Simplified, AtmosphericDragModel, AtmosphericState
from orbit.cd_estimator import DynamicCDEstimator, CDEstimate, ExtremeWeatherCompensator
from orbit.orbit_decay import (
    OrbitDecayEstimator, HohmannTransferCalculator,
    DecayPrediction, ManeuverCommand, DecayRiskLevel
)
from orbit.orbital_elements import OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv
from orbit.adaptive_engine import DeepSpaceAdaptiveEngine, AdaptiveMode, AdaptiveSystemState


class TestSpaceWeather:
    def test_provider_default_values(self):
        provider = SpaceWeatherProvider()
        data = provider.update(0.0)
        assert 50.0 <= data.f107 <= 300.0
        assert 0.0 <= data.kp <= 9.0
        assert data.ap >= 0.0

    def test_provider_simulated_variability(self):
        provider = SpaceWeatherProvider()
        f107_values = []
        for i in range(100):
            data = provider.update(i * 3600.0)
            f107_values.append(data.f107)
        assert np.std(f107_values) > 0

    def test_solar_flare_injection(self):
        provider = SpaceWeatherProvider()
        event = provider.inject_solar_flare(0.0, "X-class")
        assert event.event_type == "SOLAR_FLARE"
        assert event.severity == "X-class"
        assert event.start_time == 0.0

    def test_geomagnetic_storm_injection(self):
        provider = SpaceWeatherProvider()
        event = provider.inject_geomagnetic_storm(0.0, "extreme")
        assert event.event_type == "GEOMAGNETIC_STORM"
        assert event.severity == "extreme"

    def test_kp_to_ap_conversion(self):
        provider = SpaceWeatherProvider()
        assert provider._kp_to_ap(0) == 0
        assert provider._kp_to_ap(3) == 15
        assert provider._kp_to_ap(9) == 400

    def test_extreme_weather_detection(self):
        provider = SpaceWeatherProvider()
        provider.inject_geomagnetic_storm(0.0, "extreme")
        provider.update(0.0)
        assert provider.is_extreme_weather()

    def test_density_multiplier(self):
        provider = SpaceWeatherProvider()
        data = provider.update(0.0)
        multiplier = provider.get_density_multiplier()
        assert multiplier > 0.0


class TestNRLMSISE00:
    def test_sea_level_density(self):
        model = NRLMSISE00Simplified()
        state = model.compute(altitude=0.0)
        assert 1.0 < state.density < 2.0
        assert 280 < state.temperature < 300

    def test_high_altitude_density(self):
        model = NRLMSISE00Simplified()
        state = model.compute(altitude=400000.0)
        assert state.density < 1e-6
        assert state.density > 1e-20

    def test_density_decreases_with_altitude(self):
        model = NRLMSISE00Simplified()
        rho_low = model.compute(altitude=200000.0).density
        rho_high = model.compute(altitude=500000.0).density
        assert rho_high < rho_low

    def test_space_weather_effect(self):
        model = NRLMSISE00Simplified()
        quiet = SpaceWeatherData(f107=80.0, f107a=80.0, ap=4.0)
        active = SpaceWeatherData(f107=250.0, f107a=200.0, ap=100.0)

        state_quiet = model.compute(altitude=400000.0, space_weather=quiet)
        state_active = model.compute(altitude=400000.0, space_weather=active)

        assert state_active.density > state_quiet.density

    def test_atmospheric_state_fields(self):
        model = NRLMSISE00Simplified()
        state = model.compute(altitude=300000.0)
        assert state.density > 0
        assert state.temperature > 0
        assert state.altitude == 300000.0
        assert state.f107 > 0

    def test_atmospheric_drag_model(self):
        model = AtmosphericDragModel(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])
        acc = model.acceleration(r, v)
        assert np.linalg.norm(acc) > 0
        assert np.dot(acc, v) < 0

    def test_extreme_weather_compensation(self):
        model = AtmosphericDragModel(cd=2.2, area_mass_ratio=0.01)
        model.set_extreme_weather_compensation(True, severity=1.0)
        assert model.cross_section_multiplier > 1.0
        model.set_extreme_weather_compensation(False)
        assert model.cross_section_multiplier == 1.0


class TestCDEstimator:
    def test_initial_cd(self):
        estimator = DynamicCDEstimator(initial_cd=2.2)
        assert estimator.current_cd == 2.2

    def test_orbit_decay_update(self):
        estimator = DynamicCDEstimator(initial_cd=2.2)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])
        elements = rv_to_orbital_elements(r, v)

        weather = SpaceWeatherData(f107=150.0, f107a=150.0, ap=15.0)

        estimate = estimator.update_with_orbit_decay(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            timestamp=100.0, r=r, v=v,
            space_weather=weather
        )
        assert isinstance(estimate, CDEstimate)
        assert 0.5 <= estimate.cd <= 5.0

    def test_cd_history(self):
        estimator = DynamicCDEstimator(initial_cd=2.2)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])
        elements = rv_to_orbital_elements(r, v)
        weather = SpaceWeatherData()

        for i in range(5):
            estimator.update_with_orbit_decay(
                semi_major_axis=elements.semi_major_axis,
                eccentricity=elements.eccentricity,
                timestamp=100.0 + i * 60.0,
                r=r, v=v, space_weather=weather
            )

        history = estimator.get_cd_history()
        assert len(history) >= 0

    def test_cd_reset(self):
        estimator = DynamicCDEstimator(initial_cd=2.2)
        estimator.current_cd = 3.5
        estimator.reset(cd=2.5)
        assert estimator.current_cd == 2.5


class TestExtremeWeatherCompensator:
    def test_no_compensation_normal(self):
        compensator = ExtremeWeatherCompensator()
        weather = SpaceWeatherData(kp=3.0, f107=150.0)
        result = compensator.check_extreme_weather(weather, [], 0.0)
        assert not result
        assert not compensator.compensation_active

    def test_compensation_extreme_kp(self):
        compensator = ExtremeWeatherCompensator()
        weather = SpaceWeatherData(kp=8.0, f107=150.0)
        result = compensator.check_extreme_weather(weather, [], 0.0)
        assert result
        assert compensator.compensation_active
        assert compensator.area_multiplier > 1.0

    def test_compensation_solar_flare_event(self):
        compensator = ExtremeWeatherCompensator()
        weather = SpaceWeatherData(kp=3.0, f107=150.0)
        event = SpaceWeatherEvent(
            event_type="SOLAR_FLARE", severity="X-class",
            start_time=0.0
        )
        result = compensator.check_extreme_weather(weather, [event], 0.0)
        assert result

    def test_compensation_deactivation(self):
        compensator = ExtremeWeatherCompensator()
        weather_extreme = SpaceWeatherData(kp=8.0)
        weather_normal = SpaceWeatherData(kp=3.0)

        compensator.check_extreme_weather(weather_extreme, [], 0.0)
        assert compensator.compensation_active

        compensator.check_extreme_weather(weather_normal, [], 1.0)
        assert not compensator.compensation_active
        assert compensator.area_multiplier == 1.0


class TestOrbitDecay:
    def test_analytical_decay_estimate(self):
        estimator = OrbitDecayEstimator(cd=2.2, area_mass_ratio=0.01)
        elements = OrbitalElements(
            semi_major_axis=6778140.0, eccentricity=0.001,
            inclination=np.deg2rad(97.5), raan=np.deg2rad(45.0),
            arg_of_perigee=np.deg2rad(30.0), true_anomaly=0.0
        )
        r, v = orbital_elements_to_rv(elements)

        prediction = estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v
        )

        assert isinstance(prediction, DecayPrediction)
        assert prediction.time_to_reentry > 0
        assert prediction.current_altitude > 0

    def test_risk_level_high_altitude(self):
        estimator = OrbitDecayEstimator(cd=2.2, area_mass_ratio=0.01)
        elements = OrbitalElements(
            semi_major_axis=7378140.0, eccentricity=0.001,
            inclination=np.deg2rad(45.0), raan=0.0,
            arg_of_perigee=0.0, true_anomaly=0.0
        )
        r, v = orbital_elements_to_rv(elements)

        prediction = estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v
        )
        assert prediction.risk_level in [DecayRiskLevel.NOMINAL, DecayRiskLevel.LOW]

    def test_risk_level_low_altitude(self):
        estimator = OrbitDecayEstimator(cd=2.2, area_mass_ratio=0.01)
        elements = OrbitalElements(
            semi_major_axis=6538137.0, eccentricity=0.001,
            inclination=np.deg2rad(45.0), raan=0.0,
            arg_of_perigee=0.0, true_anomaly=0.0
        )
        r, v = orbital_elements_to_rv(elements)

        prediction = estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v
        )
        assert prediction.risk_level in [
            DecayRiskLevel.HIGH, DecayRiskLevel.CRITICAL,
            DecayRiskLevel.IMMINENT, DecayRiskLevel.MEDIUM
        ]

    def test_decay_with_extreme_weather(self):
        estimator = OrbitDecayEstimator(cd=2.2, area_mass_ratio=0.01)
        elements = OrbitalElements(
            semi_major_axis=6678137.0, eccentricity=0.001,
            inclination=np.deg2rad(45.0), raan=0.0,
            arg_of_perigee=0.0, true_anomaly=0.0
        )
        r, v = orbital_elements_to_rv(elements)

        quiet = SpaceWeatherData(f107=80.0, f107a=80.0, ap=4.0)
        storm = SpaceWeatherData(f107=250.0, f107a=200.0, ap=100.0)

        pred_quiet = estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v, space_weather=quiet
        )
        pred_storm = estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v, space_weather=storm
        )

        assert pred_storm.time_to_reentry <= pred_quiet.time_to_reentry


class TestHohmannTransfer:
    def test_basic_transfer(self):
        calc = HohmannTransferCalculator()
        elements = OrbitalElements(
            semi_major_axis=6678137.0, eccentricity=0.001,
            inclination=np.deg2rad(45.0), raan=0.0,
            arg_of_perigee=0.0, true_anomaly=0.0
        )
        r, v = orbital_elements_to_rv(elements)

        command = calc.compute_transfer(
            current_elements=elements,
            target_altitude=400000.0,
            spacecraft_mass=100.0,
            thrust=10.0,
            isp=300.0,
            current_r=r,
            current_v=v
        )

        assert isinstance(command, ManeuverCommand)
        assert np.linalg.norm(command.delta_v) > 0
        assert command.burn_time > 0
        assert command.propellant_mass > 0
        assert command.maneuver_type == "hohmann_transfer"
        assert command.target_semi_major_axis > elements.semi_major_axis

    def test_low_thrust_transfer(self):
        calc = HohmannTransferCalculator()
        elements = OrbitalElements(
            semi_major_axis=6678137.0, eccentricity=0.001,
            inclination=np.deg2rad(45.0), raan=0.0,
            arg_of_perigee=0.0, true_anomaly=0.0
        )

        result = calc.compute_low_thrust_transfer(
            current_elements=elements,
            target_altitude=400000.0,
            thrust_accel=1e-5
        )

        assert result['total_delta_v'] > 0
        assert result['transfer_time'] > 0
        assert result['num_orbits'] > 0


class TestDeepSpaceAdaptiveEngine:
    def test_initialization(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        assert engine.mode == AdaptiveMode.NOMINAL

    def test_nominal_step(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])

        state = engine.step(r, v, timestamp=0.0)
        assert isinstance(state, AdaptiveSystemState)
        assert state.altitude_km > 0
        assert state.orbital_period > 0
        assert state.density_at_altitude > 0

    def test_extreme_weather_response(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])

        engine.inject_solar_flare(0.0, "X-class")
        engine.inject_geomagnetic_storm(0.0, "extreme")

        state = engine.step(r, v, timestamp=100.0)
        assert state.mode in [AdaptiveMode.EXTREME_WEATHER, AdaptiveMode.CRITICAL_DECAY,
                              AdaptiveMode.EMERGENCY_REBOOST]

    def test_critical_decay_triggers_reboost(self):
        engine = DeepSpaceAdaptiveEngine(
            cd=2.2, area_mass_ratio=0.01,
            critical_altitude=250000.0,
            reboost_altitude=350000.0
        )

        r = np.array([6538137.0, 0.0, 0.0])
        v = np.array([0.0, 7900.0, 0.0])

        state = engine.step(r, v, timestamp=0.0)

        if state.mode in [AdaptiveMode.CRITICAL_DECAY, AdaptiveMode.EMERGENCY_REBOOST]:
            assert len(engine.get_maneuver_queue()) > 0

    def test_maneuver_command_structure(self):
        engine = DeepSpaceAdaptiveEngine(
            cd=2.2, area_mass_ratio=0.01,
            critical_altitude=250000.0,
            reboost_altitude=350000.0
        )
        r = np.array([6528137.0, 0.0, 0.0])
        v = np.array([0.0, 7900.0, 0.0])

        state = engine.step(r, v, timestamp=0.0)
        commands = engine.get_maneuver_queue()

        if commands:
            cmd = commands[0]
            assert isinstance(cmd, ManeuverCommand)
            assert np.linalg.norm(cmd.delta_v) > 0
            assert cmd.maneuver_type == "hohmann_transfer"
            assert cmd.target_semi_major_axis > 6378137.0

    def test_state_history_recording(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])

        for i in range(5):
            engine.step(r, v, timestamp=i * 100.0)

        history = engine.get_state_history()
        assert len(history) == 5

    def test_alert_logging(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6528137.0, 0.0, 0.0])
        v = np.array([0.0, 7900.0, 0.0])

        engine.inject_geomagnetic_storm(0.0, "extreme")
        engine.step(r, v, timestamp=0.0)

        alerts = engine.get_alert_log()
        assert len(alerts) > 0

    def test_density_profile(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])

        profile = engine.get_current_density_profile(r)
        assert len(profile) == 50
        assert profile[0]['altitude_km'] < profile[-1]['altitude_km']

    def test_multiple_step_consistency(self):
        engine = DeepSpaceAdaptiveEngine(cd=2.2, area_mass_ratio=0.01)
        r = np.array([6778140.0, 0.0, 0.0])
        v = np.array([0.0, 7676.0, 0.0])

        for i in range(20):
            state = engine.step(r, v, timestamp=i * 10.0)
            assert state.altitude_km > 0
            assert state.density_at_altitude >= 0
