from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class AttitudeQuaternion(BaseModel):
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    timestamp: int = 0


class GPSMeasurement(BaseModel):
    prn: int = 0
    pseudorange: float = 0.0
    carrier_phase: float = 0.0
    doppler: float = 0.0
    timestamp: int = 0


class TelemetryRequest(BaseModel):
    data: str = Field(..., description="Base64 encoded telemetry data stream")


class TelemetryResponse(BaseModel):
    timestamp: int = 0
    attitude: Optional[AttitudeQuaternion] = None
    gps_measurements: List[GPSMeasurement] = []
    housekeeping: List[float] = []
    is_valid: bool = False
    corrected_errors: int = 0


class OrbitalElementsRequest(BaseModel):
    semi_major_axis: float = Field(..., description="Semi-major axis in meters")
    eccentricity: float = Field(..., description="Eccentricity (0-1)")
    inclination: float = Field(..., description="Inclination in radians")
    raan: float = Field(..., description="Right ascension of ascending node in radians")
    arg_of_perigee: float = Field(..., description="Argument of perigee in radians")
    true_anomaly: float = Field(..., description="True anomaly in radians")
    epoch: float = 0.0


class OrbitalElementsResponse(BaseModel):
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_of_perigee: float
    true_anomaly: float
    epoch: float


class PropagationRequest(BaseModel):
    initial_elements: OrbitalElementsRequest
    duration: float = Field(..., description="Propagation duration in seconds")
    step: float = 1.0
    use_j2: bool = True
    use_drag: bool = True


class PropagationResponse(BaseModel):
    time: List[float]
    positions: List[List[float]]
    velocities: List[List[float]]
    elements: List[OrbitalElementsResponse]
    event_times: Optional[List[float]] = None


class StateVector(BaseModel):
    position: List[float] = Field(..., description="Position vector in ECI frame [x, y, z] meters")
    velocity: List[float] = Field(..., description="Velocity vector in ECI frame [vx, vy, vz] m/s")
    timestamp: float = 0.0


class EKFMeasurement(BaseModel):
    position: Optional[List[float]] = None
    gps_pseudoranges: Optional[List[GPSMeasurement]] = None
    timestamp: float
    position_noise: Optional[float] = 1.0


class EKFResponse(BaseModel):
    position: List[float]
    velocity: List[float]
    timestamp: float
    elements: Optional[OrbitalElementsResponse] = None
    position_covariance: Optional[List[List[float]]] = None


class SystemStatus(BaseModel):
    telemetry_frames_processed: int = 0
    valid_frames: int = 0
    total_corrected_errors: int = 0
    bytes_processed: int = 0
    ekf_initialized: bool = False
    last_update_time: Optional[float] = None
    connected_clients: int = 0


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: float
