import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from .orbital_elements import OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv
from .orbit_propagator import OrbitPropagator
from .space_weather import SpaceWeatherData
from .nrlmsise00 import NRLMSISE00Simplified


class DecayRiskLevel(Enum):
    NOMINAL = "nominal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMINENT = "imminent"


@dataclass
class DecayPrediction:
    time_to_reentry: float
    time_to_critical_altitude: float
    critical_altitude: float
    current_altitude: float
    semi_major_axis_decay_rate: float
    risk_level: DecayRiskLevel
    confidence: float
    prediction_method: str
    timestamp: float
    scenario_params: dict = field(default_factory=dict)


@dataclass
class ManeuverCommand:
    delta_v: np.ndarray
    burn_time: float
    thrust_magnitude: float
    isp: float
    propellant_mass: float
    maneuver_type: str
    target_semi_major_axis: float
    estimated_new_orbital_period: float
    risk_mitigation: float
    priority: int
    timestamp: float
    execute_immediately: bool = False


class OrbitDecayEstimator:
    def __init__(self, cd: float = 2.2, area_mass_ratio: float = 0.01,
                 critical_altitude: float = 200000.0):
        self.cd = cd
        self.area_mass_ratio = area_mass_ratio
        self.critical_altitude = critical_altitude
        self.mu = 3.986004418e14
        self.R_EARTH = 6378137.0

        self.nrlmsise = NRLMSISE00Simplified()
        self._decay_history: List[Tuple[float, float]] = []

    def estimate_time_to_reentry(self, elements: OrbitalElements,
                                  r: np.ndarray, v: np.ndarray,
                                  space_weather: Optional[SpaceWeatherData] = None,
                                  method: str = "analytical",
                                  cd_override: Optional[float] = None,
                                  area_multiplier: float = 1.0) -> DecayPrediction:
        current_altitude = elements.semi_major_axis * (1 - elements.eccentricity) - self.R_EARTH

        cd_effective = cd_override if cd_override is not None else self.cd
        area_ratio_effective = self.area_mass_ratio * area_multiplier

        if method == "numerical":
            t_reentry, t_critical = self._numerical_propagation(
                elements, space_weather, cd_effective, area_ratio_effective
            )
            prediction_method = "numerical_integration"
        else:
            t_reentry, t_critical = self._analytical_estimate(
                elements, r, v, space_weather, cd_effective, area_ratio_effective
            )
            prediction_method = "analytical_bate"

        da_dt = self._compute_decay_rate(
            elements, r, v, space_weather, cd_effective, area_ratio_effective
        )

        risk_level = self._assess_risk(t_reentry, current_altitude, da_dt)

        confidence = self._estimate_confidence(current_altitude, space_weather)

        return DecayPrediction(
            time_to_reentry=t_reentry,
            time_to_critical_altitude=t_critical,
            critical_altitude=self.critical_altitude,
            current_altitude=current_altitude,
            semi_major_axis_decay_rate=da_dt,
            risk_level=risk_level,
            confidence=confidence,
            prediction_method=prediction_method,
            timestamp=0.0,
            scenario_params={
                'cd': cd_effective,
                'area_mass_ratio': area_ratio_effective,
                'f107': space_weather.f107 if space_weather else 150.0,
                'kp': space_weather.kp if space_weather else 3.0,
            }
        )

    def _analytical_estimate(self, elements: OrbitalElements,
                              r: np.ndarray, v: np.ndarray,
                              space_weather: Optional[SpaceWeatherData],
                              cd: float, area_mass_ratio: float) -> Tuple[float, float]:
        a = elements.semi_major_axis
        e = elements.eccentricity
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.R_EARTH

        lat = np.rad2deg(np.arcsin(r[2] / r_mag))

        atmo = self.nrlmsise.compute(
            altitude=altitude, latitude=lat, space_weather=space_weather
        )
        rho = atmo.density

        v_mag = np.linalg.norm(v)
        n = np.sqrt(self.mu / a ** 3)
        p = a * (1 - e ** 2)

        da_dt = -2 * np.pi * cd * area_mass_ratio * rho * a ** 2 * v_mag / p

        if abs(da_dt) < 1e-12:
            return 1e10, 1e10

        current_periapsis = a * (1 - e) - self.R_EARTH
        critical_periapsis = self.critical_altitude + self.R_EARTH
        target_a = critical_periapsis / (1 - e)

        delta_a_critical = a - target_a

        if delta_a_critical > 0:
            t_critical = abs(delta_a_critical / da_dt)
        else:
            t_critical = 0.0

        reentry_altitude = 100000.0
        reentry_periapsis = reentry_altitude + self.R_EARTH
        reentry_a = reentry_periapsis / (1 - e)
        delta_a_reentry = a - reentry_a

        if delta_a_reentry > 0:
            t_reentry = abs(delta_a_reentry / da_dt)
        else:
            t_reentry = 0.0

        if space_weather is not None:
            density_mult = 1.0
            if space_weather.kp >= 8:
                density_mult *= 5.0
            elif space_weather.kp >= 7:
                density_mult *= 3.0
            elif space_weather.kp >= 6:
                density_mult *= 2.0
            elif space_weather.kp >= 5:
                density_mult *= 1.5
            f107_factor = 1.0 + 0.005 * (space_weather.f107 - 150)
            density_mult *= max(0.5, min(3.0, f107_factor))
            t_reentry /= density_mult
            t_critical /= density_mult

        return t_reentry, t_critical

    def _numerical_propagation(self, elements: OrbitalElements,
                                space_weather: Optional[SpaceWeatherData],
                                cd: float, area_mass_ratio: float) -> Tuple[float, float]:
        propagator = OrbitPropagator(use_j2=True, use_drag=True)
        propagator.atmospheric_drag.cd = cd
        propagator.atmospheric_drag.area_mass_ratio = area_mass_ratio

        r, v = orbital_elements_to_rv(elements)

        t_reentry = 0.0
        t_critical = 0.0
        found_critical = False

        dt = 60.0
        max_steps = 100000

        current_a = elements.semi_major_axis
        current_e = elements.eccentricity

        for step in range(max_steps):
            r, v = propagator.step(r, v, dt)

            new_elements = rv_to_orbital_elements(r, v)
            current_a = new_elements.semi_major_axis
            current_e = new_elements.eccentricity

            periapsis = current_a * (1 - current_e) - self.R_EARTH

            if not found_critical and periapsis <= self.critical_altitude:
                t_critical = (step + 1) * dt
                found_critical = True

            if periapsis <= 100000.0:
                t_reentry = (step + 1) * dt
                break

            t_reentry = (step + 1) * dt
        else:
            t_reentry = max_steps * dt
            if not found_critical:
                t_critical = max_steps * dt

        return t_reentry, t_critical

    def _compute_decay_rate(self, elements: OrbitalElements,
                             r: np.ndarray, v: np.ndarray,
                             space_weather: Optional[SpaceWeatherData],
                             cd: float, area_mass_ratio: float) -> float:
        a = elements.semi_major_axis
        e = elements.eccentricity
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.R_EARTH

        lat = np.rad2deg(np.arcsin(r[2] / r_mag))

        atmo = self.nrlmsise.compute(
            altitude=altitude, latitude=lat, space_weather=space_weather
        )
        rho = atmo.density

        v_mag = np.linalg.norm(v)
        p = a * (1 - e ** 2)

        da_dt = -2 * np.pi * cd * area_mass_ratio * rho * a ** 2 * v_mag / p

        return da_dt

    def _assess_risk(self, time_to_reentry: float, altitude: float,
                      decay_rate: float) -> DecayRiskLevel:
        altitude_km = altitude / 1000.0

        if altitude_km < 150 or time_to_reentry < 86400:
            return DecayRiskLevel.IMMINENT
        elif altitude_km < 200 or time_to_reentry < 7 * 86400:
            return DecayRiskLevel.CRITICAL
        elif altitude_km < 250 or time_to_reentry < 30 * 86400:
            return DecayRiskLevel.HIGH
        elif altitude_km < 300 or time_to_reentry < 90 * 86400:
            return DecayRiskLevel.MEDIUM
        elif altitude_km < 400 or time_to_reentry < 365 * 86400:
            return DecayRiskLevel.LOW
        else:
            return DecayRiskLevel.NOMINAL

    def _estimate_confidence(self, altitude: float,
                              space_weather: Optional[SpaceWeatherData]) -> float:
        altitude_km = altitude / 1000.0

        if altitude_km < 200:
            base_confidence = 0.9
        elif altitude_km < 400:
            base_confidence = 0.8
        elif altitude_km < 600:
            base_confidence = 0.7
        else:
            base_confidence = 0.5

        if space_weather is not None:
            if space_weather.kp > 5:
                base_confidence *= 0.8
            if space_weather.f107 > 200:
                base_confidence *= 0.9

        return min(0.99, max(0.3, base_confidence))

    def lifetime_sensitivity_analysis(self, elements: OrbitalElements,
                                       r: np.ndarray, v: np.ndarray,
                                       space_weather: Optional[SpaceWeatherData] = None) -> dict:
        baseline = self.estimate_time_to_reentry(elements, r, v, space_weather)

        low_cd = self.estimate_time_to_reentry(
            elements, r, v, space_weather, cd_override=self.cd * 0.7
        )
        high_cd = self.estimate_time_to_reentry(
            elements, r, v, space_weather, cd_override=self.cd * 1.5
        )

        results = {
            'baseline_lifetime_days': baseline.time_to_reentry / 86400,
            'low_cd_lifetime_days': low_cd.time_to_reentry / 86400,
            'high_cd_lifetime_days': high_cd.time_to_reentry / 86400,
            'current_altitude_km': baseline.current_altitude / 1000,
            'decay_rate_m_per_day': -baseline.semi_major_axis_decay_rate * 86400,
            'risk_level': baseline.risk_level.value,
        }

        return results


class HohmannTransferCalculator:
    def __init__(self):
        self.mu = 3.986004418e14

    def compute_transfer(self, current_elements: OrbitalElements,
                         target_altitude: float,
                         spacecraft_mass: float = 100.0,
                         thrust: float = 10.0,
                         isp: float = 300.0,
                         current_r: Optional[np.ndarray] = None,
                         current_v: Optional[np.ndarray] = None) -> ManeuverCommand:
        r_initial = current_elements.semi_major_axis
        r_target = target_altitude + 6378137.0

        a_transfer = (r_initial + r_target) / 2.0

        v_initial = np.sqrt(self.mu / r_initial)
        v_transfer_1 = np.sqrt(self.mu * (2 / r_initial - 1 / a_transfer))
        v_transfer_2 = np.sqrt(self.mu * (2 / r_target - 1 / a_transfer))
        v_final = np.sqrt(self.mu / r_target)

        delta_v1 = abs(v_transfer_1 - v_initial)
        delta_v2 = abs(v_final - v_transfer_2)
        total_delta_v = delta_v1 + delta_v2

        g0 = 9.80665
        propellant_mass = spacecraft_mass * (1 - np.exp(-total_delta_v / (isp * g0)))

        burn_time_1 = spacecraft_mass * delta_v1 / thrust
        burn_time_2 = spacecraft_mass * delta_v2 / thrust
        total_burn_time = burn_time_1 + burn_time_2

        if current_r is not None and current_v is not None:
            r_mag = np.linalg.norm(current_r)
            v_mag = np.linalg.norm(current_v)
            v_unit = current_v / v_mag
            delta_v_vec = v_unit * delta_v1
        else:
            delta_v_vec = np.array([total_delta_v, 0.0, 0.0])

        transfer_time = np.pi * np.sqrt(a_transfer ** 3 / self.mu)

        new_period = 2 * np.pi * np.sqrt(r_target ** 3 / self.mu)

        current_altitude = r_initial - 6378137.0
        risk_mitigation = min(1.0, (target_altitude - current_altitude) / 200000.0)

        priority = 1
        if target_altitude - current_altitude > 100000:
            priority = 3
        elif target_altitude - current_altitude > 50000:
            priority = 2

        return ManeuverCommand(
            delta_v=delta_v_vec,
            burn_time=total_burn_time,
            thrust_magnitude=thrust,
            isp=isp,
            propellant_mass=propellant_mass,
            maneuver_type="hohmann_transfer",
            target_semi_major_axis=r_target,
            estimated_new_orbital_period=new_period,
            risk_mitigation=risk_mitigation,
            priority=priority,
            timestamp=0.0,
            execute_immediately=False
        )

    def compute_phase_angle(self, current_elements: OrbitalElements,
                             target_raan: float,
                             target_arg_of_perigee: float) -> float:
        delta_raan = target_raan - current_elements.raan
        delta_arg = target_arg_of_perigee - current_elements.arg_of_perigee

        phase_angle = np.sqrt(delta_raan ** 2 + delta_arg ** 2)
        return phase_angle

    def compute_low_thrust_transfer(self, current_elements: OrbitalElements,
                                     target_altitude: float,
                                     thrust_accel: float = 1e-5) -> dict:
        r_initial = current_elements.semi_major_axis
        r_target = target_altitude + 6378137.0
        delta_r = r_target - r_initial

        v_initial = np.sqrt(self.mu / r_initial)
        total_delta_v = np.sqrt(self.mu / r_target) - v_initial

        transfer_time = abs(total_delta_v / thrust_accel)

        return {
            'total_delta_v': abs(total_delta_v),
            'transfer_time': transfer_time,
            'num_orbits': transfer_time / (2 * np.pi * np.sqrt(r_initial ** 3 / self.mu)),
            'average_thrust_accel': thrust_accel,
        }
