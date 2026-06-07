import numpy as np
from dataclasses import dataclass
from typing import Optional
from .space_weather import SpaceWeatherData


@dataclass
class AtmosphericState:
    density: float
    temperature: float
    pressure: float
    mean_mass: float
    altitude: float
    latitude: float
    longitude: float
    f107: float
    f107a: float
    ap: float


class NRLMSISE00Simplified:
    def __init__(self):
        self._R_EARTH = 6378137.0
        self._g0 = 9.80665
        self._M_AVG = 28.9644e-3
        self._R_GAS = 8.31432
        self._gamma = -6.5e-3

        self._altitude_bands = [
            (0.0, 11000.0, -6.5e-3, 288.15, 101325.0),
            (11000.0, 20000.0, 0.0, 216.65, 22632.1),
            (20000.0, 32000.0, 1.0e-3, 216.65, 5474.89),
            (32000.0, 47000.0, 2.8e-3, 228.65, 868.02),
            (47000.0, 51000.0, 0.0, 270.65, 110.91),
            (51000.0, 71000.0, -2.8e-3, 270.65, 66.94),
            (71000.0, 84852.0, -2.0e-3, 214.65, 3.96),
        ]

        self._thermosphere_params = {
            'T_inf_base': 1000.0,
            'T_120': 360.0,
            's': 0.01,
            'gamma_prime': 0.01,
        }

    def compute(self, altitude: float, latitude: float = 45.0,
                longitude: float = 0.0, doy: int = 180,
                lst: float = 12.0,
                space_weather: Optional[SpaceWeatherData] = None) -> AtmosphericState:

        if space_weather is not None:
            f107 = space_weather.f107
            f107a = space_weather.f107a
            ap = space_weather.ap
        else:
            f107 = 150.0
            f107a = 150.0
            ap = 15.0

        if altitude < 86000:
            density, temperature, pressure = self._troposphere_stratosphere(altitude)
        else:
            density, temperature, pressure = self._thermosphere(
                altitude, f107, f107a, ap, doy, lst, latitude)

        density = self._apply_space_weather_corrections(
            density, altitude, f107, f107a, ap, doy, lst, latitude)

        mean_mass = self._compute_mean_mass(altitude)

        return AtmosphericState(
            density=density,
            temperature=temperature,
            pressure=pressure,
            mean_mass=mean_mass,
            altitude=altitude,
            latitude=latitude,
            longitude=longitude,
            f107=f107,
            f107a=f107a,
            ap=ap
        )

    def _troposphere_stratosphere(self, altitude: float):
        for h_start, h_end, gamma, T_start, P_start in self._altitude_bands:
            if h_start <= altitude < h_end:
                h_diff = altitude - h_start

                if abs(gamma) < 1e-10:
                    T = T_start
                    P = P_start * np.exp(-self._g0 * self._M_AVG * h_diff / (self._R_GAS * T_start))
                else:
                    T = T_start + gamma * h_diff
                    P = P_start * (T / T_start) ** (-self._g0 * self._M_AVG / (self._R_GAS * gamma))

                rho = P * self._M_AVG / (self._R_GAS * T)
                return rho, T, P

        return 1e-20, 200.0, 1e-5

    def _thermosphere(self, altitude: float, f107: float, f107a: float,
                      ap: float, doy: int, lst: float, latitude: float):
        T_inf = self._compute_exospheric_temp(f107, f107a, ap, doy)

        z_120 = 120000.0
        T_120 = 360.0

        if altitude <= z_120:
            T = self._troposphere_stratosphere(min(altitude, 85000))[1]
        else:
            T = self._jb_temperature_profile(altitude, T_inf, T_120, z_120)

        rho = self._thermosphere_density(altitude, T, T_inf, f107, f107a, ap)
        P = rho * self._R_GAS * T / self._M_AVG

        return rho, T, P

    def _compute_exospheric_temp(self, f107: float, f107a: float,
                                  ap: float, doy: int) -> float:
        T_inf = 383.0 + 3.3 * f107a + 1.8 * (f107 - f107a)

        ap_effect = 65.0 * np.exp(-0.12 * ap) + 0.016 * ap
        T_inf += ap_effect

        seasonal = 20.0 * np.sin(2 * np.pi * doy / 365.25 - np.pi / 2)
        T_inf += seasonal

        return T_inf

    def _jb_temperature_profile(self, z: float, T_inf: float, T_120: float, z_120: float) -> float:
        s = 0.01
        gamma_prime = 0.01

        z_diff = z - z_120

        T = T_inf - (T_inf - T_120) * np.exp(-s * z_diff)

        if z < 300000:
            lapse_correction = gamma_prime * z_diff * np.exp(-z_diff / 50000)
            T += lapse_correction

        return T

    def _thermosphere_density(self, z: float, T: float, T_inf: float,
                               f107: float, f107a: float, ap: float) -> float:
        z_km = z / 1000.0

        if z_km <= 200:
            log_rho_coeffs = [
                -16.7087, 4.6725e-2, -8.4617e-5, 9.3806e-8,
                -5.8835e-11, 2.1048e-14, -4.0981e-18, 3.3347e-22
            ]
            log_rho = 0.0
            for i, coeff in enumerate(log_rho_coeffs):
                log_rho += coeff * z_km ** i

            f107_correction = 1.0 + 0.008 * (f107a - 150) / 150
            if f107 > 200:
                f107_correction += 0.005 * (f107 - 200) / 100

            ap_correction = 1.0
            if ap > 30:
                ap_correction = 1.0 + 0.01 * (ap - 30) / 20

            rho = 10 ** log_rho * f107_correction * ap_correction
        else:
            rho_200 = 3.614e-10
            H = self._R_GAS * T / (self._M_AVG * self._g0)
            scale_height_km = H / 1000.0
            rho = rho_200 * np.exp(-(z_km - 200) / scale_height_km)

            f107_correction = 1.0 + 0.008 * (f107a - 150) / 150
            if f107 > 200:
                f107_correction += 0.005 * (f107 - 200) / 100
            rho *= f107_correction

            ap_correction = 1.0
            if ap > 30:
                ap_correction = 1.0 + 0.01 * (ap - 30) / 20
            rho *= ap_correction

        if z_km < 500:
            geom_factor = 1.0 + 0.1 * np.exp(-(z_km - 200) / 100)
            rho *= geom_factor

        return max(rho, 1e-25)

    def _apply_space_weather_corrections(self, rho: float, altitude: float,
                                          f107: float, f107a: float,
                                          ap: float, doy: int,
                                          lst: float, lat: float) -> float:
        z_km = altitude / 1000.0

        if z_km > 100:
            semiannual = 1.0 + 0.1 * np.sin(4 * np.pi * doy / 365.25)
            rho *= semiannual

        if 200 < z_km < 600:
            diurnal = 1.0 + 0.15 * np.sin(2 * np.pi * (lst - 14) / 24)
            rho *= diurnal

        lat_rad = np.deg2rad(lat)
        if z_km > 300:
            lat_variation = 1.0 + 0.1 * np.cos(lat_rad) ** 2
            rho *= lat_variation

        if ap > 50 and z_km < 500:
            storm_enhancement = 1.0 + 0.5 * np.exp(-(z_km - 250) / 100)
            rho *= storm_enhancement

        return rho

    def _compute_mean_mass(self, altitude: float) -> float:
        z_km = altitude / 1000.0

        if z_km < 100:
            return 28.9644e-3
        elif z_km < 200:
            fraction = (z_km - 100) / 100
            return 28.9644e-3 * (1 - fraction) + 20.0e-3 * fraction
        elif z_km < 500:
            fraction = (z_km - 200) / 300
            return 20.0e-3 * (1 - fraction) + 4.0e-3 * fraction
        else:
            return 1.0e-3


class AtmosphericDragModel:
    def __init__(self, cd: float = 2.2, area_mass_ratio: float = 0.01):
        self.nrlmsise = NRLMSISE00Simplified()
        self.base_cd = cd
        self.area_mass_ratio = area_mass_ratio
        self.dynamic_cd = cd
        self.cross_section_multiplier = 1.0

    def acceleration(self, r: np.ndarray, v: np.ndarray,
                     space_weather: SpaceWeatherData = None,
                     timestamp: float = None) -> np.ndarray:
        r_mag = np.linalg.norm(r)
        altitude = r_mag - 6378137.0

        v_rel = v
        v_mag = np.linalg.norm(v_rel)

        if v_mag < 1e-10:
            return np.zeros(3)

        doy = 180
        lst = 12.0
        lat = np.rad2deg(np.arcsin(r[2] / r_mag))

        atmo = self.nrlmsise.compute(
            altitude=altitude,
            latitude=lat,
            space_weather=space_weather
        )

        rho = atmo.density

        cd_effective = self.dynamic_cd * self.cross_section_multiplier

        acc_mag = -0.5 * rho * cd_effective * self.area_mass_ratio * v_mag
        acc = acc_mag * v_rel

        return acc

    def update_cd(self, estimated_cd: float):
        self.dynamic_cd = estimated_cd

    def get_density(self, r: np.ndarray, space_weather: SpaceWeatherData = None) -> float:
        r_mag = np.linalg.norm(r)
        altitude = r_mag - 6378137.0

        lat = np.rad2deg(np.arcsin(r[2] / r_mag))

        atmo = self.nrlmsise.compute(
            altitude=altitude,
            latitude=lat,
            space_weather=space_weather
        )

        return atmo.density

    def set_extreme_weather_compensation(self, enabled: bool, severity: float = 1.0):
        if enabled:
            self.cross_section_multiplier = 1.0 + 0.5 * severity
        else:
            self.cross_section_multiplier = 1.0
