import numpy as np
from typing import List, Optional, Tuple
import struct

try:
    from telemetry_core import CCSDSFrameSync, FrameResult
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False


class FrameParser:
    def __init__(self, sync_threshold: int = 2):
        self.sync_threshold = sync_threshold
        if HAS_CPP_EXTENSION:
            self._sync = CCSDSFrameSync()
            self._sync.sync_threshold = sync_threshold
        else:
            self._buffer = bytearray()

        self.ASM_SYNC_WORD = 0x1ACFFC1D
        self.ASM_LENGTH = 4
        self.MIN_FRAME_LENGTH = 64
        self.MAX_FRAME_LENGTH = 2048

    def _hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def _extract_uint32_msb(self, data: bytes, offset: int) -> int:
        if offset + 4 > len(data):
            return 0
        return struct.unpack('>I', data[offset:offset+4])[0]

    def _validate_frame_header(self, data: bytes, offset: int) -> Optional[dict]:
        if offset + 10 > len(data):
            return None

        frame_version = (data[offset + 4] >> 5) & 0x07
        spacecraft_id = ((data[offset + 4] & 0x1F) << 8) | data[offset + 5]
        virtual_channel_id = (data[offset + 6] >> 2) & 0x3F

        frame_length = ((data[offset + 6] & 0x03) << 8 | data[offset + 7]) + 1

        if frame_length < self.MIN_FRAME_LENGTH or frame_length > self.MAX_FRAME_LENGTH:
            return None

        if offset + frame_length > len(data):
            return None

        payload_start = offset + 10
        payload_end = offset + frame_length

        return {
            'found': True,
            'offset': offset,
            'frame_length': frame_length,
            'spacecraft_id': spacecraft_id,
            'virtual_channel_id': virtual_channel_id,
            'frame_version': frame_version,
            'payload': bytes(data[payload_start:payload_end]),
            'bit_errors': 0
        }

    def sync_frames(self, data_stream: bytes) -> List[dict]:
        if HAS_CPP_EXTENSION:
            data_list = list(data_stream)
            results = self._sync.sync_frames(data_list)
            return [
                {
                    'found': r.found,
                    'offset': r.offset,
                    'frame_length': r.frame_length,
                    'spacecraft_id': r.spacecraft_id,
                    'virtual_channel_id': r.virtual_channel_id,
                    'frame_version': r.frame_version,
                    'payload': bytes(r.payload),
                    'bit_errors': r.bit_errors
                }
                for r in results
            ]

        results = []
        if len(data_stream) < self.ASM_LENGTH + self.MIN_FRAME_LENGTH:
            return results

        for i in range(len(data_stream) - self.ASM_LENGTH - self.MIN_FRAME_LENGTH + 1):
            candidate = self._extract_uint32_msb(data_stream, i)
            errors = self._hamming_distance(candidate, self.ASM_SYNC_WORD)

            if errors <= self.sync_threshold:
                result = self._validate_frame_header(data_stream, i)
                if result:
                    result['bit_errors'] = errors
                    results.append(result)
                    i += result['frame_length'] - 1

        return results

    def process_stream(self, data_chunk: bytes) -> List[dict]:
        self._buffer.extend(data_chunk)
        frames = self.sync_frames(bytes(self._buffer))

        if frames:
            last_offset = frames[-1]['offset'] + frames[-1]['frame_length']
            self._buffer = self._buffer[last_offset:]

        return frames

    def reset(self):
        self._buffer.clear()
