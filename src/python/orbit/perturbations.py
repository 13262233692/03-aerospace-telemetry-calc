import numpy as np
from typing import Optional
from .orbital_elements import OrbitalConstants


class J2Perturbation:
    def __init__(self, j2: Optional[float] = None, radius: Optional[float] = None, mu: Optional[float] = None):
        self.j2 = j2 if j2 is not None else OrbitalConstants.EARTH_J2
        self.radius = radius if radius is not None else OrbitalConstants.EARTH_RADIUS
        self.mu = mu if mu is not None else OrbitalConstants.EARTH_MU

    def acceleration(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        r_mag = np.linalg.norm(r)

        if r_mag < 1e-10:
            return np.zeros(3)

        x, y, z = r
        z_over_r2 = (z / r_mag) ** 2

        factor = -1.5 * self.j2 * self.mu * (self.radius ** 2) / (r_mag ** 5)

        ax = factor * x * (1 - 5 * z_over_r2)
        ay = factor * y * (1 - 5 * z_over_r2)
        az = factor * z * (3 - 5 * z_over_r2)

        return np.array([ax, ay, az])

    def orbital_element_rates(self, elements) -> np.ndarray:
        a = elements.semi_major_axis
        e = elements.eccentricity
        i = elements.inclination
        raan = elements.raan
        arg_of_perigee = elements.arg_of_perigee
        nu = elements.true_anomaly

        p = a * (1 - e**2)
        n = np.sqrt(self.mu / a**3)

        factor = 0.75 * self.j2 * n * (self.radius / p)**2

        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)
        cos_argp_nu = np.cos(arg_of_perigee + nu)
        sin_argp_nu = np.sin(arg_of_perigee + nu)
        cos_2argp_nu = np.cos(2 * arg_of_perigee + 2 * nu)

        da = 0.0
        de = factor * (1 - e**2) * sin_i**2 * np.sin(2 * arg_of_perigee + 2 * nu)
        di = factor * e * cos_i * np.sin(2 * arg_of_perigee + 2 * nu)
        draan = -2 * factor * cos_i * (1 + (e * cos_nu) / (1 + e * cos_nu))
        dargp = factor * ((5 * cos_i**2 - 1) / (1 - e**2) +
                          2 * sin_i**2 * cos_2argp_nu / (1 + e * cos_nu) +
                          e * cos_nu * (5 * cos_i**2 - 1) / (1 + e * cos_nu))
        dnu = n + factor * np.sqrt(1 - e**2) * ((5 * cos_i**2 - 1) / (1 - e**2) +
                                                2 * sin_i**2 * cos_2argp_nu / (1 + e * cos_nu))

        return np.array([da, de, di, draan, dargp, dnu])


class AtmosphericDrag:
    def __init__(self, cd: float = 2.2, area_mass_ratio: float = 0.01,
                 radius: Optional[float] = None, mu: Optional[float] = None):
        self.cd = cd
        self.area_mass_ratio = area_mass_ratio
        self.radius = radius if radius is not None else OrbitalConstants.EARTH_RADIUS
        self.mu = mu if mu is not None else OrbitalConstants.EARTH_MU
        self.earth_omega = np.array([0, 0, OrbitalConstants.EARTH_OMEGA])

    def _exponential_atmosphere(self, altitude: float) -> float:
        if altitude < 200000:
            h0 = 0.0
            rho0 = 1.225
            H = 8500.0
        elif altitude < 300000:
            h0 = 200000.0
            rho0 = 2.0e-9
            H = 35000.0
        elif altitude < 500000:
            h0 = 300000.0
            rho0 = 1.0e-10
            H = 50000.0
        elif altitude < 800000:
            h0 = 500000.0
            rho0 = 5.0e-12
            H = 60000.0
        else:
            h0 = 800000.0
            rho0 = 1.0e-14
            H = 70000.0

        return rho0 * np.exp(-(altitude - h0) / H)

    def density(self, r: np.ndarray) -> float:
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.radius
        return self._exponential_atmosphere(altitude)

    def acceleration(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:
            return np.zeros(3)

        v_rel = v - np.cross(self.earth_omega, r)
        v_rel_mag = np.linalg.norm(v_rel)

        if v_rel_mag < 1e-10:
            return np.zeros(3)

        rho = self.density(r)
        factor = -0.5 * self.cd * self.area_mass_ratio * rho * v_rel_mag

        return factor * v_rel


class ThirdBodyPerturbation:
    def __init__(self, body_mu: float, body_position_func):
        self.body_mu = body_mu
        self.body_position_func = body_position_func

    def acceleration(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        r_body = np.asarray(self.body_position_func(t), dtype=np.float64)

        r_sc_body = r_body - r
        r_sc_mag = np.linalg.norm(r)
        r_body_mag = np.linalg.norm(r_body)
        r_sc_body_mag = np.linalg.norm(r_sc_body)

        if r_sc_body_mag < 1e-10 or r_body_mag < 1e-10:
            return np.zeros(3)

        a = self.body_mu * (r_sc_body / r_sc_body_mag**3 - r_body / r_body_mag**3)
        return a


class SolarRadiationPressure:
    def __init__(self, area_mass_ratio: float = 0.01, reflectivity: float = 1.0):
        self.area_mass_ratio = area_mass_ratio
        self.reflectivity = reflectivity
        self.solar_constant = 1361.0
        self.c = 299792458.0

    def acceleration(self, r_sun: np.ndarray) -> np.ndarray:
        r_sun = np.asarray(r_sun, dtype=np.float64)
        r_mag = np.linalg.norm(r_sun)

        if r_mag < 1e-10:
            return np.zeros(3)

        p = self.solar_constant / self.c
        factor = -p * self.area_mass_ratio * (1 + self.reflectivity) / r_mag

        return factor * r_sun
