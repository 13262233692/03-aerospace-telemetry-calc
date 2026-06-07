import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import time


@dataclass
class SpaceWeatherData:
    timestamp: float = 0.0
    f107: float = 150.0
    f107a: float = 150.0
    kp: float = 3.0
    ap: float = 15.0
    solar_wind_speed: float = 400.0
    solar_wind_density: float = 5.0
    bz: float = 0.0
    dst: float = 0.0


@dataclass
class SpaceWeatherEvent:
    event_type: str
    severity: str
    start_time: float
    end_time: Optional[float] = None
    parameters: Dict[str, float] = field(default_factory=dict)


class SpaceWeatherProvider:
    def __init__(self, use_real_data: bool = False):
        self.use_real_data = use_real_data
        self.current_data = SpaceWeatherData()
        self.event_history: List[SpaceWeatherEvent] = []
        self._simulation_time = 0.0
        self._base_f107 = 150.0
        self._base_kp = 3.0

    def update(self, timestamp: float) -> SpaceWeatherData:
        self._simulation_time = timestamp

        if self.use_real_data:
            self._fetch_real_data()
        else:
            self._simulate_data(timestamp)

        return self.current_data

    def _simulate_data(self, timestamp: float):
        day_of_year = (datetime.fromtimestamp(timestamp, tz=None).timetuple().tm_yday
                       if timestamp > 0 else 180)

        seasonal_variation = 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)
        solar_cycle_variation = 0.2 * np.sin(2 * np.pi * timestamp / (11 * 365 * 86400))

        f107_noise = np.random.normal(0, 10.0)
        self.current_data.f107 = max(50.0, min(300.0,
            self._base_f107 * (1 + seasonal_variation + solar_cycle_variation) + f107_noise))

        self.current_data.f107a = 0.9 * self.current_data.f107a + 0.1 * self.current_data.f107

        kp_noise = np.random.normal(0, 0.5)
        self.current_data.kp = max(0.0, min(9.0,
            self._base_kp + seasonal_variation * 3 + kp_noise))

        self.current_data.ap = self._kp_to_ap(self.current_data.kp)

        self.current_data.solar_wind_speed = 400.0 + np.random.normal(0, 50.0)
        self.current_data.solar_wind_density = max(1.0, 5.0 + np.random.normal(0, 2.0))
        self.current_data.bz = np.random.normal(0, 3.0)
        self.current_data.dst = -10.0 + np.random.normal(0, 20.0)

        self.current_data.timestamp = timestamp

    def _kp_to_ap(self, kp: float) -> float:
        kp_to_ap_map = {
            0: 0, 0.3: 2, 0.7: 3, 1: 4, 1.3: 5, 1.7: 6,
            2: 7, 2.3: 9, 2.7: 12, 3: 15, 3.3: 18, 3.7: 22,
            4: 27, 4.3: 32, 4.7: 39, 5: 48, 5.3: 56, 5.7: 67,
            6: 80, 6.3: 94, 6.7: 111, 7: 132, 7.3: 154, 7.7: 179,
            8: 207, 8.3: 236, 8.7: 300, 9: 400
        }

        kp_floor = int(kp * 10) / 10
        if kp_floor in kp_to_ap_map:
            return kp_to_ap_map[kp_floor]
        return 15.0

    def _fetch_real_data(self):
        pass

    def inject_solar_flare(self, timestamp: float, magnitude: str = "M-class"):
        params = {}
        if magnitude == "X-class":
            flare_f107_increase = 50.0
            params["intensity"] = 1e-4
        elif magnitude == "M-class":
            flare_f107_increase = 20.0
            params["intensity"] = 1e-5
        else:
            flare_f107_increase = 5.0
            params["intensity"] = 1e-6

        self._base_f107 += flare_f107_increase

        event = SpaceWeatherEvent(
            event_type="SOLAR_FLARE",
            severity=magnitude,
            start_time=timestamp,
            parameters=params
        )
        self.event_history.append(event)
        return event

    def inject_geomagnetic_storm(self, timestamp: float, intensity: str = "moderate"):
        params = {}
        if intensity == "extreme":
            kp_jump = 5.0
            params["dst_min"] = -200.0
        elif intensity == "strong":
            kp_jump = 4.0
            params["dst_min"] = -100.0
        elif intensity == "moderate":
            kp_jump = 2.5
            params["dst_min"] = -50.0
        else:
            kp_jump = 1.0
            params["dst_min"] = -30.0

        self._base_kp = min(9.0, self._base_kp + kp_jump)

        event = SpaceWeatherEvent(
            event_type="GEOMAGNETIC_STORM",
            severity=intensity,
            start_time=timestamp,
            parameters=params
        )
        self.event_history.append(event)
        return event

    def get_active_events(self, timestamp: float) -> List[SpaceWeatherEvent]:
        active = []
        for event in self.event_history:
            if event.end_time is None or event.end_time > timestamp:
                active.append(event)
        return active

    def is_extreme_weather(self) -> bool:
        if self.current_data.kp >= 7:
            return True
        if self.current_data.f107 >= 250:
            return True
        if self.current_data.dst <= -100:
            return True
        return False

    def get_density_multiplier(self) -> float:
        multiplier = 1.0

        if self.current_data.kp >= 8:
            multiplier *= 5.0
        elif self.current_data.kp >= 7:
            multiplier *= 3.0
        elif self.current_data.kp >= 6:
            multiplier *= 2.0
        elif self.current_data.kp >= 5:
            multiplier *= 1.5

        f107_factor = 1.0 + 0.005 * (self.current_data.f107 - 150)
        multiplier *= max(0.5, min(3.0, f107_factor))

        return multiplier
