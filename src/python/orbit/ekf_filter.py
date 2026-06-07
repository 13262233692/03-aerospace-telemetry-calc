import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .orbital_elements import OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv
from .orbit_propagator import OrbitPropagator


@dataclass
class EKFState:
    x: np.ndarray
    P: np.ndarray
    timestamp: float

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]


class ExtendedKalmanFilter:
    def __init__(self, propagator: OrbitPropagator,
                 process_noise: Optional[np.ndarray] = None,
                 initial_state: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None):
        self.propagator = propagator
        self.state_dim = 6
        self.measurement_dim = 3

        if process_noise is None:
            self.Q = np.eye(self.state_dim) * 1e-6
        else:
            self.Q = process_noise.copy()

        if initial_state is None:
            self.state = EKFState(
                x=np.zeros(self.state_dim),
                P=np.eye(self.state_dim) * 1e6,
                timestamp=0.0
            )
        else:
            self.state = EKFState(
                x=initial_state.copy(),
                P=initial_covariance if initial_covariance is not None else np.eye(self.state_dim) * 1e6,
                timestamp=0.0
            )

        self.R_default = np.eye(self.measurement_dim) * 1e3

    def initialize_from_elements(self, elements: OrbitalElements,
                                  position_uncertainty: float = 100.0,
                                  velocity_uncertainty: float = 1.0,
                                  timestamp: float = 0.0):
        r, v = orbital_elements_to_rv(elements)
        self.state.x = np.concatenate([r, v])

        P = np.eye(self.state_dim)
        P[:3, :3] *= position_uncertainty ** 2
        P[3:6, 3:6] *= velocity_uncertainty ** 2
        self.state.P = P
        self.state.timestamp = timestamp

    def initialize_from_rv(self, r: np.ndarray, v: np.ndarray,
                            position_uncertainty: float = 100.0,
                            velocity_uncertainty: float = 1.0,
                            timestamp: float = 0.0):
        self.state.x = np.concatenate([r, v])

        P = np.eye(self.state_dim)
        P[:3, :3] *= position_uncertainty ** 2
        P[3:6, 3:6] *= velocity_uncertainty ** 2
        self.state.P = P
        self.state.timestamp = timestamp

    def predict(self, dt: float, timestamp: float) -> EKFState:
        if dt <= 0:
            return self.state

        r = self.state.position
        v = self.state.velocity

        r_new, v_new = self.propagator.step(r, v, dt)

        F = self.propagator.get_state_transition_matrix(r, v, dt)

        x_new = np.concatenate([r_new, v_new])
        P_new = F @ self.state.P @ F.T + self.Q

        self.state = EKFState(
            x=x_new,
            P=P_new,
            timestamp=timestamp
        )

        return self.state

    def h_function(self, x: np.ndarray) -> np.ndarray:
        return x[:3]

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[:3, :3] = np.eye(3)
        return H

    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None,
               timestamp: Optional[float] = None) -> Tuple[EKFState, np.ndarray]:

        z = np.asarray(measurement, dtype=np.float64)

        if measurement_noise is None:
            R = self.R_default
        else:
            R = np.asarray(measurement_noise, dtype=np.float64)

        if timestamp is not None and timestamp > self.state.timestamp:
            dt = timestamp - self.state.timestamp
            self.predict(dt, timestamp)

        H = self.H_jacobian(self.state.x)
        h_x = self.h_function(self.state.x)

        y = z - h_x

        S = H @ self.state.P @ H.T + R

        K = self.state.P @ H.T @ np.linalg.inv(S)

        x_new = self.state.x + K @ y

        I = np.eye(self.state_dim)
        P_new = (I - K @ H) @ self.state.P

        if timestamp is not None:
            self.state = EKFState(x=x_new, P=P_new, timestamp=timestamp)
        else:
            self.state = EKFState(x=x_new, P=P_new, timestamp=self.state.timestamp)

        return self.state, y

    def process_measurement(self, measurement: np.ndarray,
                            measurement_timestamp: float,
                            measurement_noise: Optional[np.ndarray] = None) -> EKFState:
        if measurement_timestamp > self.state.timestamp:
            dt = measurement_timestamp - self.state.timestamp
            self.predict(dt, measurement_timestamp)

        state, innovation = self.update(measurement, measurement_noise, measurement_timestamp)
        return state

    def process_gps_pseudorange(self, pseudoranges: List[dict],
                                 satellite_positions: dict,
                                 measurement_timestamp: float,
                                 measurement_noise: float = 1.0) -> EKFState:
        if measurement_timestamp > self.state.timestamp:
            dt = measurement_timestamp - self.state.timestamp
            self.predict(dt, measurement_timestamp)

        n_meas = len(pseudoranges)
        if n_meas == 0:
            return self.state

        z = np.zeros(n_meas)
        h_x = np.zeros(n_meas)
        H = np.zeros((n_meas, self.state_dim))

        r_sc = self.state.position

        for i, pr in enumerate(pseudoranges):
            prn = pr['prn']
            if prn in satellite_positions:
                r_sat = satellite_positions[prn]
                range_est = np.linalg.norm(r_sat - r_sc)
                h_x[i] = range_est
                z[i] = pr['pseudorange']

                if range_est > 1e-10:
                    los = (r_sc - r_sat) / range_est
                    H[i, :3] = los

        R = np.eye(n_meas) * measurement_noise ** 2

        y = z - h_x
        S = H @ self.state.P @ H.T + R

        try:
            K = self.state.P @ H.T @ np.linalg.inv(S)

            x_new = self.state.x + K @ y

            I = np.eye(self.state_dim)
            P_new = (I - K @ H) @ self.state.P

            self.state = EKFState(x=x_new, P=P_new, timestamp=measurement_timestamp)
        except np.linalg.LinAlgError:
            pass

        return self.state

    def get_orbital_elements(self) -> Optional[OrbitalElements]:
        try:
            elem = rv_to_orbital_elements(self.state.position, self.state.velocity)
            elem.epoch = self.state.timestamp
            return elem
        except Exception:
            return None

    def get_position_covariance(self) -> np.ndarray:
        return self.state.P[:3, :3]

    def get_velocity_covariance(self) -> np.ndarray:
        return self.state.P[3:6, 3:6]
