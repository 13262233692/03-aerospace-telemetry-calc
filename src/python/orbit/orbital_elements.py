import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class OrbitalElements:
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_of_perigee: float
    true_anomaly: float
    epoch: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            self.raan,
            self.arg_of_perigee,
            self.true_anomaly
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, epoch: float = 0.0) -> 'OrbitalElements':
        return cls(
            semi_major_axis=arr[0],
            eccentricity=arr[1],
            inclination=arr[2],
            raan=arr[3],
            arg_of_perigee=arr[4],
            true_anomaly=arr[5],
            epoch=epoch
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'semi_major_axis': self.semi_major_axis,
            'eccentricity': self.eccentricity,
            'inclination': self.inclination,
            'raan': self.raan,
            'arg_of_perigee': self.arg_of_perigee,
            'true_anomaly': self.true_anomaly,
            'epoch': self.epoch
        }

    def __repr__(self) -> str:
        return (f"OrbitalElements(a={self.semi_major_axis:.2f}m, e={self.eccentricity:.6f}, "
                f"i={np.degrees(self.inclination):.2f}°, Ω={np.degrees(self.raan):.2f}°, "
                f"ω={np.degrees(self.arg_of_perigee):.2f}°, ν={np.degrees(self.true_anomaly):.2f}°)")


class OrbitalConstants:
    EARTH_MU = 3.986004418e14
    EARTH_RADIUS = 6378137.0
    EARTH_J2 = 1.082635854e-3
    EARTH_OMEGA = 7.292115146706979e-5
    EARTH_FLATTENING = 1.0 / 298.257223563


def rv_to_orbital_elements(r: np.ndarray, v: np.ndarray, mu: float = OrbitalConstants.EARTH_MU) -> OrbitalElements:
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)

    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    specific_energy = v_mag**2 / 2 - mu / r_mag

    if abs(e - 1.0) > 1e-10:
        a = -mu / (2 * specific_energy)
    else:
        a = float('inf')

    i = np.arccos(h[2] / h_mag)

    if n_mag != 0:
        raan = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0

    if n_mag != 0 and e > 1e-15:
        arg_of_perigee = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            arg_of_perigee = 2 * np.pi - arg_of_perigee
    else:
        arg_of_perigee = 0.0

    if e > 1e-15:
        true_anomaly = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0.0

    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=i,
        raan=raan,
        arg_of_perigee=arg_of_perigee,
        true_anomaly=true_anomaly
    )


def orbital_elements_to_rv(elements: OrbitalElements, mu: float = OrbitalConstants.EARTH_MU) -> Tuple[np.ndarray, np.ndarray]:
    a = elements.semi_major_axis
    e = elements.eccentricity
    i = elements.inclination
    raan = elements.raan
    arg_of_perigee = elements.arg_of_perigee
    nu = elements.true_anomaly

    p = a * (1 - e**2)

    r_pqw = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0.0
    ])

    v_pqw = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0.0
    ])

    R_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])

    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])

    R_argp = np.array([
        [np.cos(arg_of_perigee), -np.sin(arg_of_perigee), 0],
        [np.sin(arg_of_perigee), np.cos(arg_of_perigee), 0],
        [0, 0, 1]
    ])

    R = R_raan @ R_i @ R_argp

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    return r_eci, v_eci


def mean_anomaly_to_eccentric(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    if e < 0.8:
        E = M
    else:
        E = np.pi

    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break

    return E


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))


def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                          np.sqrt(1 + e) * np.cos(nu / 2))
