import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

from .orbital_elements import (
    OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv,
    OrbitalConstants
)
from .perturbations import J2Perturbation, AtmosphericDrag


@dataclass
class PropagationResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    elements: List[OrbitalElements]
    event_times: Optional[np.ndarray] = None


class OrbitPropagator:
    def __init__(self, mu: Optional[float] = None, use_j2: bool = True, use_drag: bool = True):
        self.mu = mu if mu is not None else OrbitalConstants.EARTH_MU
        self.use_j2 = use_j2
        self.use_drag = use_drag

        self.j2_perturbation = J2Perturbation() if use_j2 else None
        self.drag_perturbation = AtmosphericDrag() if use_drag else None

        self.min_radius = OrbitalConstants.EARTH_RADIUS * 0.9

    def _two_body_acceleration(self, r: np.ndarray) -> np.ndarray:
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:
            return np.zeros(3)
        return -self.mu * r / r_mag**3

    def _total_acceleration(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        acc = self._two_body_acceleration(r)

        if self.j2_perturbation:
            acc += self.j2_perturbation.acceleration(r)

        if self.drag_perturbation:
            acc += self.drag_perturbation.acceleration(r, v)

        return acc

    def _equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]

        if np.linalg.norm(r) < self.min_radius:
            return np.zeros(6)

        a = self._total_acceleration(r, v)

        return np.concatenate([v, a])

    def _ground_strike_event(self, t: float, y: np.ndarray) -> float:
        r = y[:3]
        return np.linalg.norm(r) - self.min_radius

    def propagate(self, initial_r: np.ndarray, initial_v: np.ndarray,
                  t_span: Tuple[float, float], dt: float = 1.0,
                  method: str = 'RK45', rtol: float = 1e-8, atol: float = 1e-10,
                  events: Optional[List[Callable]] = None) -> PropagationResult:

        initial_r = np.asarray(initial_r, dtype=np.float64)
        initial_v = np.asarray(initial_v, dtype=np.float64)

        y0 = np.concatenate([initial_r, initial_v])

        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        all_events = [self._ground_strike_event]
        if events:
            all_events.extend(events)

        sol = solve_ivp(
            self._equations_of_motion,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            events=all_events,
            dense_output=True
        )

        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")

        time = sol.t
        position = sol.y[:3, :].T
        velocity = sol.y[3:6, :].T

        elements_list = []
        for i in range(len(time)):
            try:
                elem = rv_to_orbital_elements(position[i], velocity[i], self.mu)
                elem.epoch = time[i]
                elements_list.append(elem)
            except Exception:
                elements_list.append(None)

        event_times = None
        if sol.t_events and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
            event_times = sol.t_events[0]

        return PropagationResult(
            time=time,
            position=position,
            velocity=velocity,
            elements=elements_list,
            event_times=event_times
        )

    def propagate_elements(self, initial_elements: OrbitalElements,
                           t_span: Tuple[float, float], dt: float = 1.0,
                           **kwargs) -> PropagationResult:
        r, v = orbital_elements_to_rv(initial_elements, self.mu)
        return self.propagate(r, v, t_span, dt, **kwargs)

    def step(self, r: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        result = self.propagate(r, v, (0, dt), dt=dt)
        return result.position[-1], result.velocity[-1]

    def get_state_transition_matrix(self, r: np.ndarray, v: np.ndarray, dt: float,
                                     perturbation: float = 1e-6) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        n = 6
        stm = np.eye(n)

        for i in range(n):
            y_perturbed = np.zeros(n)
            y_perturbed[i] = perturbation

            r0_p = r + y_perturbed[:3]
            v0_p = v + y_perturbed[3:]

            r1, v1 = self.step(r0_p, v0_p, dt)
            y1 = np.concatenate([r1, v1])

            r0_n = r - y_perturbed[:3]
            v0_n = v - y_perturbed[3:]

            r2, v2 = self.step(r0_n, v0_n, dt)
            y2 = np.concatenate([r2, v2])

            stm[:, i] = (y1 - y2) / (2 * perturbation)

        return stm
