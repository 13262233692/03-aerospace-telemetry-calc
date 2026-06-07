import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque

from .nrlmsise00 import NRLMSISE00Simplified, AtmosphericState
from .space_weather import SpaceWeatherData, SpaceWeatherEvent


@dataclass
class CDEstimate:
    cd: float
    uncertainty: float
    timestamp: float
    method: str
    raw_measurements: dict = field(default_factory=dict)


class DynamicCDEstimator:
    def __init__(self, initial_cd: float = 2.2, window_size: int = 100):
        self.current_cd = initial_cd
        self.cd_uncertainty = 0.3
        self.window_size = window_size

        self._cd_history = deque(maxlen=window_size)
        self._residual_history = deque(maxlen=window_size)
        self._measurement_history = deque(maxlen=window_size)

        self._ekf_state = np.array([initial_cd])
        self._ekf_P = np.array([[0.1]])
        self._ekf_Q = np.array([[1e-4]])
        self._ekf_R = np.array([[0.01]])

        self._knockout_factor = 1.0
        self._ram_wake_factor = 1.0
        self._molecular_flow_factor = 1.0

        self._last_semi_major_axis = None
        self._last_measurement_time = None

        self.nrlmsise = NRLMSISE00Simplified()

    def update_with_orbit_decay(self, semi_major_axis: float,
                                 eccentricity: float,
                                 timestamp: float,
                                 r: np.ndarray,
                                 v: np.ndarray,
                                 space_weather: SpaceWeatherData,
                                 area_mass_ratio: float = 0.01) -> CDEstimate:
        estimate_method = "model_based"

        if self._last_semi_major_axis is not None and self._last_measurement_time is not None:
            dt = timestamp - self._last_measurement_time
            if dt > 60.0:
                da_measured = semi_major_axis - self._last_semi_major_axis

                expected_decay = self._compute_expected_decay(
                    semi_major_axis, eccentricity, r, v,
                    space_weather, area_mass_ratio, dt
                )

                if abs(expected_decay) > 1e-6:
                    cd_measured = self.current_cd * (da_measured / expected_decay)
                    cd_measured = np.clip(cd_measured, 0.5, 5.0)

                    self._ekf_predict()
                    self._ekf_update(cd_measured)

                    self.current_cd = float(self._ekf_state[0])
                    self.cd_uncertainty = float(np.sqrt(self._ekf_P[0, 0]))

                    self._cd_history.append((timestamp, self.current_cd))
                    self._residual_history.append((timestamp, da_measured - expected_decay))
                    estimate_method = "orbit_fit_ekf"

        self._last_semi_major_axis = semi_major_axis
        self._last_measurement_time = timestamp

        corrected_cd = self._apply_physical_corrections(
            self.current_cd, r, v, space_weather
        )

        return CDEstimate(
            cd=corrected_cd,
            uncertainty=self.cd_uncertainty,
            timestamp=timestamp,
            method=estimate_method,
            raw_measurements={
                'semi_major_axis': semi_major_axis,
                'eccentricity': eccentricity,
            }
        )

    def update_with_acceleration_measurement(self, measured_accel: np.ndarray,
                                              predicted_accel: np.ndarray,
                                              r: np.ndarray,
                                              v: np.ndarray,
                                              timestamp: float,
                                              rho: float,
                                              area_mass_ratio: float = 0.01) -> CDEstimate:
        v_mag = np.linalg.norm(v)
        if v_mag < 1e-10 or rho < 1e-20:
            return CDEstimate(
                cd=self.current_cd,
                uncertainty=self.cd_uncertainty,
                timestamp=timestamp,
                method="model_based"
            )

        v_unit = v / v_mag

        measured_drag = np.dot(measured_accel, v_unit)
        predicted_drag = np.dot(predicted_accel, v_unit)

        if abs(predicted_drag) > 1e-15:
            cd_ratio = measured_drag / predicted_drag
            cd_measured = self.current_cd * cd_ratio
            cd_measured = np.clip(cd_measured, 0.5, 5.0)

            self._ekf_predict()
            self._ekf_update(cd_measured)

            self.current_cd = float(self._ekf_state[0])
            self.cd_uncertainty = float(np.sqrt(self._ekf_P[0, 0]))

            self._cd_history.append((timestamp, self.current_cd))
            self._measurement_history.append((timestamp, cd_measured))

        corrected_cd = self._apply_physical_corrections(
            self.current_cd, r, v, None
        )

        return CDEstimate(
            cd=corrected_cd,
            uncertainty=self.cd_uncertainty,
            timestamp=timestamp,
            method="accel_ekf"
        )

    def _ekf_predict(self):
        self._ekf_P = self._ekf_P + self._ekf_Q

    def _ekf_update(self, measurement: float):
        H = np.array([[1.0]])

        y = measurement - H @ self._ekf_state
        S = H @ self._ekf_P @ H.T + self._ekf_R
        K = self._ekf_P @ H.T @ np.linalg.inv(S)

        self._ekf_state = self._ekf_state + K @ y
        I = np.eye(1)
        self._ekf_P = (I - K @ H) @ self._ekf_P

    def _compute_expected_decay(self, semi_major_axis: float,
                                 eccentricity: float,
                                 r: np.ndarray,
                                 v: np.ndarray,
                                 space_weather: SpaceWeatherData,
                                 area_mass_ratio: float,
                                 dt: float) -> float:
        mu = 3.986004418e14
        a = semi_major_axis
        e = eccentricity

        r_mag = np.linalg.norm(r)
        altitude = r_mag - 6378137.0
        lat = np.rad2deg(np.arcsin(r[2] / r_mag))

        atmo = self.nrlmsise.compute(altitude=altitude, latitude=lat, space_weather=space_weather)
        rho = atmo.density

        v_mag = np.linalg.norm(v)

        n = np.sqrt(mu / a ** 3)
        p = a * (1 - e ** 2)

        da_dt = -2 * np.pi * self.current_cd * area_mass_ratio * rho * a ** 2 * v_mag / p

        expected_da = da_dt * dt

        return expected_da

    def _apply_physical_corrections(self, cd: float, r: np.ndarray, v: np.ndarray,
                                     space_weather: Optional[SpaceWeatherData]) -> float:
        corrected = cd

        r_mag = np.linalg.norm(r)
        altitude = r_mag - 6378137.0

        corrected *= self._molecular_flow_correction(altitude)

        v_mag = np.linalg.norm(v)
        corrected *= self._velocity_correction(v_mag)

        if space_weather is not None:
            corrected *= self._space_weather_correction(space_weather, altitude)

        return np.clip(corrected, 0.5, 5.0)

    def _molecular_flow_correction(self, altitude: float) -> float:
        z_km = altitude / 1000.0

        if z_km < 100:
            return 1.0
        elif z_km < 120:
            t = (z_km - 100) / 20
            return 1.0 + 0.1 * t
        elif z_km < 200:
            t = (z_km - 120) / 80
            return 1.1 + 0.2 * t
        else:
            return 1.3

    def _velocity_correction(self, v_mag: float) -> float:
        v_km_s = v_mag / 1000.0

        if v_km_s < 5:
            return 1.0
        elif v_km_s < 10:
            t = (v_km_s - 5) / 5
            return 1.0 - 0.05 * t
        else:
            return 0.95

    def _space_weather_correction(self, space_weather: SpaceWeatherData,
                                   altitude: float) -> float:
        z_km = altitude / 1000.0

        correction = 1.0

        if space_weather.f107 > 200 and z_km > 200:
            f107_factor = 1.0 + 0.1 * (space_weather.f107 - 200) / 100
            correction *= f107_factor

        if space_weather.ap > 50 and z_km < 500:
            ap_factor = 1.0 + 0.15 * (space_weather.ap - 50) / 100
            correction *= ap_factor

        return correction

    def get_cd_history(self) -> List[Tuple[float, float]]:
        return list(self._cd_history)

    def get_cd_trend(self) -> float:
        if len(self._cd_history) < 2:
            return 0.0

        times = np.array([t for t, _ in self._cd_history])
        cds = np.array([cd for _, cd in self._cd_history])

        if len(times) > 1 and np.std(times) > 0:
            coeffs = np.polyfit(times, cds, 1)
            return coeffs[0]
        return 0.0

    def reset(self, cd: float = 2.2):
        self.current_cd = cd
        self.cd_uncertainty = 0.3
        self._ekf_state = np.array([cd])
        self._ekf_P = np.array([[0.1]])
        self._cd_history.clear()
        self._residual_history.clear()
        self._measurement_history.clear()
        self._last_semi_major_axis = None
        self._last_measurement_time = None


class ExtremeWeatherCompensator:
    def __init__(self):
        self.compensation_active = False
        self.compensation_level = 0.0
        self.area_multiplier = 1.0
        self.cd_multiplier = 1.0
        self.event_start_time = None
        self.event_severity = 0.0

    def check_extreme_weather(self, space_weather: SpaceWeatherData,
                               active_events: List[SpaceWeatherEvent],
                               timestamp: float) -> bool:
        extreme = False
        severity = 0.0

        if space_weather.kp >= 8:
            extreme = True
            severity = max(severity, (space_weather.kp - 7) / 2)
        elif space_weather.kp >= 7:
            extreme = True
            severity = max(severity, (space_weather.kp - 6) / 2)

        if space_weather.f107 >= 250:
            extreme = True
            severity = max(severity, (space_weather.f107 - 200) / 100)

        if space_weather.dst <= -150:
            extreme = True
            severity = max(severity, abs(space_weather.dst) / 200)

        for event in active_events:
            if event.event_type in ["SOLAR_FLARE", "GEOMAGNETIC_STORM"]:
                if event.severity in ["X-class", "extreme", "strong"]:
                    extreme = True
                    severity = max(severity, 1.0)
                elif event.severity in ["M-class", "moderate"]:
                    extreme = True
                    severity = max(severity, 0.5)

        if extreme and not self.compensation_active:
            self._activate_compensation(severity, timestamp)
        elif not extreme and self.compensation_active:
            self._deactivate_compensation(timestamp)
        elif extreme and self.compensation_active:
            self.compensation_level = max(self.compensation_level, severity)
            self._update_multipliers()

        return extreme

    def _activate_compensation(self, severity: float, timestamp: float):
        self.compensation_active = True
        self.compensation_level = severity
        self.event_start_time = timestamp
        self._update_multipliers()

    def _deactivate_compensation(self, timestamp: float):
        self.compensation_active = False
        self.compensation_level = 0.0
        self.area_multiplier = 1.0
        self.cd_multiplier = 1.0
        self.event_start_time = None

    def _update_multipliers(self):
        base_area = 1.0 + 0.3 * self.compensation_level
        base_cd = 1.0 + 0.2 * self.compensation_level

        self.area_multiplier = base_area
        self.cd_multiplier = base_cd

    def get_effective_area_multiplier(self) -> float:
        return self.area_multiplier

    def get_effective_cd_multiplier(self) -> float:
        return self.cd_multiplier

    def get_compensation_status(self) -> dict:
        return {
            'active': self.compensation_active,
            'level': self.compensation_level,
            'area_multiplier': self.area_multiplier,
            'cd_multiplier': self.cd_multiplier,
            'event_start_time': self.event_start_time
        }
