import numpy as np
from typing import List, Optional, Dict, Any
import struct
import time

from .frame_parser import FrameParser
from .rs_decoder import ReedSolomonDecoder

try:
    from telemetry_core import TelemetryParser as CppTelemetryParser, TelemetryData
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False


class AttitudeQuaternion:
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, timestamp: int = 0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = timestamp

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

    def __repr__(self) -> str:
        return f"AttitudeQuaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f}, ts={self.timestamp})"


class GPSPseudorange:
    def __init__(self, prn: int = 0, pseudorange: float = 0.0, carrier_phase: float = 0.0,
                 doppler: float = 0.0, timestamp: int = 0):
        self.prn = prn
        self.pseudorange = pseudorange
        self.carrier_phase = carrier_phase
        self.doppler = doppler
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"GPSPseudorange(PRN={self.prn}, range={self.pseudorange:.3f}m, ts={self.timestamp})"


class TelemetryPacket:
    def __init__(self):
        self.timestamp: int = 0
        self.attitude: Optional[AttitudeQuaternion] = None
        self.gps_measurements: List[GPSPseudorange] = []
        self.housekeeping: List[float] = []
        self.is_valid: bool = False
        self.raw_payload: Optional[bytes] = None
        self.corrected_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'attitude': {
                'w': self.attitude.w if self.attitude else 0.0,
                'x': self.attitude.x if self.attitude else 0.0,
                'y': self.attitude.y if self.attitude else 0.0,
                'z': self.attitude.z if self.attitude else 0.0
            } if self.attitude else None,
            'gps_measurements': [
                {
                    'prn': gps.prn,
                    'pseudorange': gps.pseudorange,
                    'carrier_phase': gps.carrier_phase,
                    'doppler': gps.doppler,
                    'timestamp': gps.timestamp
                }
                for gps in self.gps_measurements
            ],
            'housekeeping': self.housekeeping,
            'is_valid': self.is_valid,
            'corrected_errors': self.corrected_errors
        }


class TelemetryProcessor:
    def __init__(self, use_rs_decode: bool = True):
        self.frame_parser = FrameParser()
        self.rs_decoder = ReedSolomonDecoder()
        self.use_rs_decode = use_rs_decode

        if HAS_CPP_EXTENSION:
            self._cpp_parser = CppTelemetryParser()

        self.stats = {
            'total_frames': 0,
            'valid_frames': 0,
            'total_corrected_errors': 0,
            'bytes_processed': 0
        }

    def _parse_double_be(self, data: bytes, offset: int) -> float:
        return struct.unpack('>d', data[offset:offset+8])[0]

    def _parse_uint64_be(self, data: bytes, offset: int) -> int:
        return struct.unpack('>Q', data[offset:offset+8])[0]

    def _parse_uint32_be(self, data: bytes, offset: int) -> int:
        return struct.unpack('>I', data[offset:offset+4])[0]

    def _parse_attitude(self, data: bytes, offset: int) -> AttitudeQuaternion:
        return AttitudeQuaternion(
            timestamp=self._parse_uint64_be(data, offset),
            w=self._parse_double_be(data, offset + 8),
            x=self._parse_double_be(data, offset + 16),
            y=self._parse_double_be(data, offset + 24),
            z=self._parse_double_be(data, offset + 32)
        )

    def _parse_gps(self, data: bytes, offset: int) -> GPSPseudorange:
        return GPSPseudorange(
            prn=self._parse_uint32_be(data, offset),
            pseudorange=self._parse_double_be(data, offset + 4),
            carrier_phase=self._parse_double_be(data, offset + 12),
            doppler=self._parse_double_be(data, offset + 20),
            timestamp=self._parse_uint64_be(data, offset + 28)
        )

    def _parse_payload_python(self, payload: bytes) -> TelemetryPacket:
        packet = TelemetryPacket()
        packet.raw_payload = payload

        if len(payload) < 48:
            return packet

        try:
            offset = 0
            packet.timestamp = self._parse_uint64_be(payload, offset)
            offset += 8

            packet.attitude = self._parse_attitude(payload, offset)
            offset += 40

            gps_count = payload[offset]
            offset += 1

            for _ in range(gps_count):
                if offset + 36 <= len(payload):
                    packet.gps_measurements.append(self._parse_gps(payload, offset))
                    offset += 36

            if offset + 1 <= len(payload):
                hk_count = payload[offset]
                offset += 1
                for _ in range(hk_count):
                    if offset + 8 <= len(payload):
                        packet.housekeeping.append(self._parse_double_be(payload, offset))
                        offset += 8

            packet.is_valid = True
        except (struct.error, IndexError):
            packet.is_valid = False

        return packet

    def _parse_payload_cpp(self, payload: bytes) -> TelemetryPacket:
        try:
            cpp_data = self._cpp_parser.parse_ccsds_payload(list(payload))
            packet = TelemetryPacket()
            packet.raw_payload = payload
            packet.timestamp = cpp_data.timestamp
            packet.is_valid = cpp_data.is_valid

            if cpp_data.attitude:
                packet.attitude = AttitudeQuaternion(
                    w=cpp_data.attitude.w,
                    x=cpp_data.attitude.x,
                    y=cpp_data.attitude.y,
                    z=cpp_data.attitude.z,
                    timestamp=cpp_data.attitude.timestamp
                )

            for gps in cpp_data.gps_measurements:
                packet.gps_measurements.append(GPSPseudorange(
                    prn=gps.prn,
                    pseudorange=gps.pseudorange,
                    carrier_phase=gps.carrier_phase,
                    doppler=gps.doppler,
                    timestamp=gps.timestamp
                ))

            packet.housekeeping = list(cpp_data.housekeeping)
            return packet
        except Exception:
            return self._parse_payload_python(payload)

    def process_payload(self, payload: bytes) -> TelemetryPacket:
        if self.use_rs_decode and len(payload) == self.rs_decoder.RS_BLOCK_LENGTH:
            decoded, corrected = self.rs_decoder.decode(payload)
            self.stats['total_corrected_errors'] += corrected
            payload = decoded

        if HAS_CPP_EXTENSION:
            packet = self._parse_payload_cpp(payload)
        else:
            packet = self._parse_payload_python(payload)

        if packet.is_valid and packet.attitude:
            packet.attitude.normalize()

        return packet

    def process_data_stream(self, data: bytes) -> List[TelemetryPacket]:
        self.stats['bytes_processed'] += len(data)

        frames = self.frame_parser.process_stream(data)
        packets = []

        for frame in frames:
            self.stats['total_frames'] += 1

            if frame['found'] and frame['payload']:
                packet = self.process_payload(frame['payload'])
                if packet.is_valid:
                    self.stats['valid_frames'] += 1
                packets.append(packet)

        return packets

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def reset_stats(self):
        self.stats = {
            'total_frames': 0,
            'valid_frames': 0,
            'total_corrected_errors': 0,
            'bytes_processed': 0
        }
