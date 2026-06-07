import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

from .space_weather import SpaceWeatherProvider, SpaceWeatherData, SpaceWeatherEvent
from .nrlmsise00 import NRLMSISE00Simplified, AtmosphericDragModel
from .cd_estimator import DynamicCDEstimator, CDEstimate, ExtremeWeatherCompensator
from .orbit_decay import (
    OrbitDecayEstimator, HohmannTransferCalculator,
    DecayPrediction, ManeuverCommand, DecayRiskLevel
)
from .orbital_elements import OrbitalElements, rv_to_orbital_elements
from .orbit_propagator import OrbitPropagator


class AdaptiveMode(Enum):
    NOMINAL = "nominal"
    ENHANCED_MONITORING = "enhanced_monitoring"
    EXTREME_WEATHER = "extreme_weather"
    CRITICAL_DECAY = "critical_decay"
    EMERGENCY_REBOOST = "emergency_reboost"


@dataclass
class AdaptiveSystemState:
    mode: AdaptiveMode
    current_cd_estimate: CDEstimate
    decay_prediction: DecayPrediction
    active_events: List[SpaceWeatherEvent]
    space_weather: SpaceWeatherData
    compensation_status: dict
    maneuver_commands: List[ManeuverCommand]
    orbital_period: float
    altitude_km: float
    density_at_altitude: float


class DeepSpaceAdaptiveEngine:
    def __init__(self, cd: float = 2.2, area_mass_ratio: float = 0.01,
                 critical_altitude: float = 200000.0,
                 reboost_altitude: float = 300000.0,
                 spacecraft_mass: float = 100.0,
                 thrust: float = 10.0,
                 isp: float = 300.0,
                 use_real_weather: bool = False):
        self.critical_altitude = critical_altitude
        self.reboost_altitude = reboost_altitude
        self.spacecraft_mass = spacecraft_mass
        self.thrust = thrust
        self.isp = isp
        self.R_EARTH = 6378137.0
        self.mu = 3.986004418e14

        self.weather_provider = SpaceWeatherProvider(use_real_data=use_real_weather)
        self.cd_estimator = DynamicCDEstimator(initial_cd=cd)
        self.weather_compensator = ExtremeWeatherCompensator()
        self.decay_estimator = OrbitDecayEstimator(
            cd=cd, area_mass_ratio=area_mass_ratio,
            critical_altitude=critical_altitude
        )
        self.hohmann_calculator = HohmannTransferCalculator()
        self.atmospheric_model = AtmosphericDragModel(cd=cd, area_mass_ratio=area_mass_ratio)

        self.mode = AdaptiveMode.NOMINAL
        self._state_history: List[AdaptiveSystemState] = []
        self._maneuver_queue: List[ManeuverCommand] = []
        self._alert_log: List[Dict] = []

    def step(self, r: np.ndarray, v: np.ndarray, timestamp: float,
             dt: float = 1.0) -> AdaptiveSystemState:
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.R_EARTH

        space_weather = self.weather_provider.update(timestamp)
        active_events = self.weather_provider.get_active_events(timestamp)

        is_extreme = self.weather_compensator.check_extreme_weather(
            space_weather, active_events, timestamp
        )

        elements = rv_to_orbital_elements(r, v)

        cd_estimate = self.cd_estimator.update_with_orbit_decay(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            timestamp=timestamp,
            r=r, v=v,
            space_weather=space_weather,
            area_mass_ratio=self.atmospheric_model.area_mass_ratio
        )

        area_multiplier = self.weather_compensator.get_effective_area_multiplier()
        cd_multiplier = self.weather_compensator.get_effective_cd_multiplier()

        decay_prediction = self.decay_estimator.estimate_time_to_reentry(
            elements=elements, r=r, v=v,
            space_weather=space_weather,
            cd_override=cd_estimate.cd,
            area_multiplier=area_multiplier * cd_multiplier
        )

        self.atmospheric_model.update_cd(cd_estimate.cd)
        self.atmospheric_model.set_extreme_weather_compensation(
            is_extreme, self.weather_compensator.compensation_level
        )

        new_mode = self._determine_mode(decay_prediction, is_extreme, altitude)
        self.mode = new_mode

        maneuver_commands = []
        if new_mode == AdaptiveMode.CRITICAL_DECAY or new_mode == AdaptiveMode.EMERGENCY_REBOOST:
            maneuver_commands = self._generate_reboost_commands(
                elements, r, v, decay_prediction, timestamp
            )
            self._maneuver_queue.extend(maneuver_commands)

            self._log_alert(
                timestamp=timestamp,
                alert_type="CRITICAL_ORBIT_DECAY",
                severity=decay_prediction.risk_level.value,
                message=f"Altitude {altitude/1000:.1f} km, Time to reentry: {decay_prediction.time_to_reentry/86400:.1f} days",
                decay_prediction=decay_prediction
            )

        if is_extreme:
            self._log_alert(
                timestamp=timestamp,
                alert_type="EXTREME_SPACE_WEATHER",
                severity="high",
                message=f"Kp={space_weather.kp:.1f}, F10.7={space_weather.f107:.1f}, Dst={space_weather.dst:.1f}",
                space_weather=space_weather
            )

        density = self.atmospheric_model.get_density(r, space_weather)
        orbital_period = 2 * np.pi * np.sqrt(elements.semi_major_axis ** 3 / self.mu)

        state = AdaptiveSystemState(
            mode=self.mode,
            current_cd_estimate=cd_estimate,
            decay_prediction=decay_prediction,
            active_events=active_events,
            space_weather=space_weather,
            compensation_status=self.weather_compensator.get_compensation_status(),
            maneuver_commands=maneuver_commands,
            orbital_period=orbital_period,
            altitude_km=altitude / 1000.0,
            density_at_altitude=density
        )

        self._state_history.append(state)
        return state

    def _determine_mode(self, decay: DecayPrediction, is_extreme: bool,
                         altitude: float) -> AdaptiveMode:
        if decay.risk_level in [DecayRiskLevel.IMMINENT, DecayRiskLevel.CRITICAL]:
            if decay.time_to_reentry < 7 * 86400:
                return AdaptiveMode.EMERGENCY_REBOOST
            return AdaptiveMode.CRITICAL_DECAY

        if is_extreme:
            return AdaptiveMode.EXTREME_WEATHER

        if decay.risk_level == DecayRiskLevel.HIGH:
            return AdaptiveMode.ENHANCED_MONITORING

        if decay.risk_level == DecayRiskLevel.MEDIUM:
            return AdaptiveMode.ENHANCED_MONITORING

        return AdaptiveMode.NOMINAL

    def _generate_reboost_commands(self, elements: OrbitalElements,
                                    r: np.ndarray, v: np.ndarray,
                                    decay: DecayPrediction,
                                    timestamp: float) -> List[ManeuverCommand]:
        commands = []

        command = self.hohmann_calculator.compute_transfer(
            current_elements=elements,
            target_altitude=self.reboost_altitude,
            spacecraft_mass=self.spacecraft_mass,
            thrust=self.thrust,
            isp=self.isp,
            current_r=r,
            current_v=v
        )

        command.timestamp = timestamp
        command.execute_immediately = (
            decay.risk_level in [DecayRiskLevel.IMMINENT, DecayRiskLevel.CRITICAL]
        )

        commands.append(command)

        safe_altitude = self.reboost_altitude + 50000
        if decay.risk_level == DecayRiskLevel.IMMINENT:
            safety_command = self.hohmann_calculator.compute_transfer(
                current_elements=elements,
                target_altitude=safe_altitude,
                spacecraft_mass=self.spacecraft_mass,
                thrust=self.thrust,
                isp=self.isp,
                current_r=r,
                current_v=v
            )
            safety_command.timestamp = timestamp
            safety_command.execute_immediately = True
            safety_command.priority = 1
            commands.append(safety_command)

        return commands

    def _log_alert(self, timestamp: float, alert_type: str, severity: str,
                    message: str, **kwargs):
        alert = {
            'timestamp': timestamp,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
        }
        alert.update(kwargs)
        self._alert_log.append(alert)

    def get_alert_log(self) -> List[Dict]:
        return self._alert_log

    def get_maneuver_queue(self) -> List[ManeuverCommand]:
        return self._maneuver_queue

    def clear_maneuver_queue(self):
        self._maneuver_queue.clear()

    def get_state_history(self) -> List[AdaptiveSystemState]:
        return self._state_history

    def get_current_density_profile(self, r: np.ndarray,
                                     alt_range: Tuple[float, float] = (100, 800),
                                     num_points: int = 50) -> List[Dict]:
        results = []
        altitudes = np.linspace(alt_range[0] * 1000, alt_range[1] * 1000, num_points)

        space_weather = self.weather_provider.current_data
        lat = np.rad2deg(np.arcsin(r[2] / np.linalg.norm(r)))

        for alt in altitudes:
            atmo = self.atmospheric_model.nrlmsise.compute(
                altitude=alt, latitude=lat, space_weather=space_weather
            )
            results.append({
                'altitude_km': alt / 1000.0,
                'density_kg_m3': atmo.density,
                'temperature_K': atmo.temperature,
            })

        return results

    def inject_solar_flare(self, timestamp: float, magnitude: str = "M-class"):
        return self.weather_provider.inject_solar_flare(timestamp, magnitude)

    def inject_geomagnetic_storm(self, timestamp: float, intensity: str = "moderate"):
        return self.weather_provider.inject_geomagnetic_storm(timestamp, intensity)
