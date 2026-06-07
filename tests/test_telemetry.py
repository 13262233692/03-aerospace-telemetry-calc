import sys
import os
import struct
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.python.telemetry.frame_parser import FrameParser
from src.python.telemetry.rs_decoder import ReedSolomonDecoder
from src.python.telemetry.telemetry_processor import TelemetryProcessor


def build_test_ccsds_frame(payload: bytes, spacecraft_id: int = 42, virtual_channel_id: int = 1) -> bytes:
    frame_version = 0
    asm = struct.pack('>I', 0x1ACFFC1D)

    header = bytearray(6)
    header[0] = (frame_version << 5) | ((spacecraft_id >> 8) & 0x1F)
    header[1] = spacecraft_id & 0xFF
    header[2] = (virtual_channel_id << 2) | 0
    frame_length = 10 + len(payload) - 1
    header[2] |= (frame_length >> 8) & 0x03
    header[3] = frame_length & 0xFF
    header[4] = 0
    header[5] = 0

    return asm + bytes(header) + payload


def build_test_payload() -> bytes:
    payload = bytearray()

    timestamp = 1234567890123456789
    payload += struct.pack('>Q', timestamp)

    att_ts = timestamp
    payload += struct.pack('>Q', att_ts)
    payload += struct.pack('>d', 0.70710678)
    payload += struct.pack('>d', 0.0)
    payload += struct.pack('>d', 0.0)
    payload += struct.pack('>d', 0.70710678)

    gps_count = 2
    payload.append(gps_count)

    for i in range(gps_count):
        payload += struct.pack('>I', 10 + i)
        payload += struct.pack('>d', 26500000.0 + i * 1000)
        payload += struct.pack('>d', 123456.789 + i)
        payload += struct.pack('>d', -1000.0 + i * 10)
        payload += struct.pack('>Q', timestamp + i)

    hk_count = 2
    payload.append(hk_count)
    payload += struct.pack('>d', 25.5)
    payload += struct.pack('>d', 3.14159)

    return bytes(payload)


class TestFrameParser:
    def test_initialization(self):
        parser = FrameParser()
        assert parser is not None
        assert parser.ASM_SYNC_WORD == 0x1ACFFC1D

    def test_sync_single_frame(self):
        parser = FrameParser()
        payload = build_test_payload()
        frame = build_test_ccsds_frame(payload)

        frames = parser.sync_frames(frame)
        assert len(frames) == 1
        assert frames[0]['found'] is True
        assert frames[0]['spacecraft_id'] == 42
        assert frames[0]['virtual_channel_id'] == 1

    def test_sync_multiple_frames(self):
        parser = FrameParser()
        payload = build_test_payload()
        frame1 = build_test_ccsds_frame(payload)
        frame2 = build_test_ccsds_frame(payload, spacecraft_id=43)

        data = frame1 + b'\x00\x00\x00' + frame2
        frames = parser.sync_frames(data)

        assert len(frames) == 2
        assert frames[0]['spacecraft_id'] == 42
        assert frames[1]['spacecraft_id'] == 43

    def test_process_stream_incremental(self):
        parser = FrameParser()
        payload = build_test_payload()
        frame = build_test_ccsds_frame(payload)

        chunks = [frame[:50], frame[50:100], frame[100:]]
        all_frames = []

        for chunk in chunks:
            frames = parser.process_stream(chunk)
            all_frames.extend(frames)

        assert len(all_frames) == 1
        assert all_frames[0]['found'] is True


class TestReedSolomonDecoder:
    def test_initialization(self):
        decoder = ReedSolomonDecoder()
        assert decoder is not None
        assert decoder.RS_BLOCK_LENGTH == 255
        assert decoder.RS_DATA_LENGTH == 223

    def test_encode_decode_no_errors(self):
        decoder = ReedSolomonDecoder()

        data = bytes(range(223))
        encoded = decoder.encode(data)

        assert len(encoded) == 255

        decoded, corrected = decoder.decode(encoded)
        assert corrected == 0
        assert decoded[:223] == data

    def test_encode_decode_with_errors(self):
        decoder = ReedSolomonDecoder()

        data = bytes(range(223))
        encoded = bytearray(decoder.encode(data))

        encoded[10] ^= 0xFF
        encoded[20] ^= 0xAA
        encoded[30] ^= 0x55

        decoded, corrected = decoder.decode(bytes(encoded))
        assert corrected >= 1
        assert decoded[:223] == data

    def test_decode_exceeding_error_capacity(self):
        decoder = ReedSolomonDecoder()

        data = bytes(range(223))
        encoded = bytearray(decoder.encode(data))

        for i in range(20):
            encoded[i * 10] ^= 0xFF

        decoded, corrected = decoder.decode(bytes(encoded))
        assert len(decoded) == 0 or decoded[:223] != data


class TestTelemetryProcessor:
    def test_initialization(self):
        processor = TelemetryProcessor()
        assert processor is not None

    def test_process_valid_payload(self):
        processor = TelemetryProcessor(use_rs_decode=False)

        payload = build_test_payload()
        packet = processor.process_payload(payload)

        assert packet.is_valid is True
        assert packet.attitude is not None
        assert abs(packet.attitude.w - 0.70710678) < 1e-6
        assert abs(packet.attitude.z - 0.70710678) < 1e-6
        assert len(packet.gps_measurements) == 2
        assert packet.gps_measurements[0].prn == 10
        assert len(packet.housekeeping) == 2

    def test_process_data_stream(self):
        processor = TelemetryProcessor(use_rs_decode=False)

        payload = build_test_payload()
        frame = build_test_ccsds_frame(payload)

        packets = processor.process_data_stream(frame)
        assert len(packets) >= 1

        stats = processor.get_stats()
        assert stats['total_frames'] >= 1
        assert stats['valid_frames'] >= 1
        assert stats['bytes_processed'] == len(frame)

    def test_attitude_normalization(self):
        processor = TelemetryProcessor(use_rs_decode=False)

        payload = bytearray(build_test_payload())
        payload[16:24] = struct.pack('>d', 2.0)
        payload[24:32] = struct.pack('>d', 0.0)
        payload[32:40] = struct.pack('>d', 0.0)
        payload[40:48] = struct.pack('>d', 0.0)

        packet = processor.process_payload(bytes(payload))
        assert packet.is_valid is True
        assert packet.attitude is not None

        norm = np.sqrt(packet.attitude.w**2 + packet.attitude.x**2 +
                       packet.attitude.y**2 + packet.attitude.z**2)
        assert abs(norm - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
