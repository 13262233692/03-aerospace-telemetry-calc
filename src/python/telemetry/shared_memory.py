import numpy as np
import struct
import time
from typing import Optional, Any, Dict
import threading

try:
    from telemetry_core import SharedMemoryChannel, LockFreeRingBuffer
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False


class SharedMemoryBuffer:
    def __init__(self, name: str, capacity: int = 1024 * 1024):
        self.name = name
        self.capacity = capacity
        self._shm = None
        self._lock = threading.Lock()

        if HAS_CPP_EXTENSION:
            self._shm = SharedMemoryChannel()
        else:
            self._buffer = None
            self._header = None

    def create(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._shm.create(self.name, self.capacity)
        else:
            self._buffer = bytearray(self.capacity + 64)
            self._header = {
                'write_counter': 0,
                'read_counter': 0,
                'data_size': 0,
                'is_valid': True
            }
            return True

    def open(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._shm.open(self.name)
        else:
            return True

    def close(self):
        if HAS_CPP_EXTENSION and self._shm:
            self._shm.close()
        else:
            self._buffer = None
            self._header = None

    def write(self, data: bytes) -> bool:
        with self._lock:
            if HAS_CPP_EXTENSION:
                return self._shm.write(data)
            else:
                if self._buffer is None or len(data) > self.capacity:
                    return False
                self._buffer[:len(data)] = data
                self._header['data_size'] = len(data)
                self._header['write_counter'] += 1
                return True

    def read(self, max_size: Optional[int] = None) -> Optional[bytes]:
        with self._lock:
            if HAS_CPP_EXTENSION:
                size = max_size if max_size else self.capacity
                data = self._shm.read(size)
                return data if data else None
            else:
                if self._buffer is None:
                    return None
                size = self._header['data_size']
                if max_size and size > max_size:
                    size = max_size
                data = bytes(self._buffer[:size])
                self._header['read_counter'] += 1
                return data

    def write_numpy(self, array: np.ndarray) -> bool:
        data = array.tobytes()
        header = struct.pack('<II', array.dtype.itemsize, array.ndim)
        header += struct.pack('<' + 'I' * array.ndim, *array.shape)
        return self.write(header + data)

    def read_numpy(self) -> Optional[np.ndarray]:
        data = self.read()
        if not data or len(data) < 8:
            return None

        try:
            itemsize, ndim = struct.unpack('<II', data[:8])
            offset = 8
            shape = struct.unpack('<' + 'I' * ndim, data[offset:offset + 4 * ndim])
            offset += 4 * ndim

            dtype = np.dtype(f'float{itemsize * 8}')
            array = np.frombuffer(data[offset:], dtype=dtype).reshape(shape)
            return array.copy()
        except Exception:
            return None

    def is_open(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._shm.is_open if self._shm else False
        else:
            return self._buffer is not None

    @property
    def write_count(self) -> int:
        if HAS_CPP_EXTENSION and self._shm:
            return self._shm.write_count
        elif self._header:
            return self._header['write_counter']
        return 0

    @property
    def read_count(self) -> int:
        if HAS_CPP_EXTENSION and self._shm:
            return self._shm.read_count
        elif self._header:
            return self._header['read_counter']
        return 0


class RingBuffer:
    def __init__(self, name: str, slot_count: int = 4096, slot_size: int = 256):
        self.name = name
        self.slot_count = slot_count
        self.slot_size = slot_size
        self._rb = None

        if HAS_CPP_EXTENSION:
            self._rb = LockFreeRingBuffer()
        else:
            self._slots = [None] * slot_count
            self._write_pos = 0
            self._read_pos = 0

    def init(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._rb.init(self.name, self.slot_count, self.slot_size)
        else:
            return True

    def enqueue(self, data: bytes) -> bool:
        if HAS_CPP_EXTENSION:
            return self._rb.enqueue(data)
        else:
            if len(data) > self.slot_size - 8:
                return False
            next_write = (self._write_pos + 1) % self.slot_count
            if next_write == self._read_pos:
                return False
            self._slots[self._write_pos] = data
            self._write_pos = next_write
            return True

    def dequeue(self) -> Optional[bytes]:
        if HAS_CPP_EXTENSION:
            data = self._rb.dequeue(self.slot_size)
            return data if data else None
        else:
            if self._read_pos == self._write_pos:
                return None
            data = self._slots[self._read_pos]
            self._read_pos = (self._read_pos + 1) % self.slot_count
            return data

    def available_slots(self) -> int:
        if HAS_CPP_EXTENSION:
            return self._rb.available_slots
        else:
            return (self._read_pos - self._write_pos - 1) % self.slot_count

    def used_slots(self) -> int:
        if HAS_CPP_EXTENSION:
            return self._rb.used_slots
        else:
            return (self._write_pos - self._read_pos) % self.slot_count

    def is_empty(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._rb.is_empty
        else:
            return self._write_pos == self._read_pos

    def is_full(self) -> bool:
        if HAS_CPP_EXTENSION:
            return self._rb.is_full
        else:
            return (self._write_pos + 1) % self.slot_count == self._read_pos


class TelemetrySharedMemory:
    def __init__(self, name: str = "aerospace_telemetry_shm"):
        self.shm = SharedMemoryBuffer(name, 65536)
        self.state_buffer = SharedMemoryBuffer(name + "_state", 1024)
        self._initialized = False

    def initialize(self, create: bool = True) -> bool:
        shm_ok = self.shm.create() if create else self.shm.open()
        state_ok = self.state_buffer.create() if create else self.state_buffer.open()
        self._initialized = shm_ok and state_ok
        return self._initialized

    def write_telemetry(self, telemetry_data: Dict[str, Any]) -> bool:
        if not self._initialized:
            return False

        try:
            import json
            data_str = json.dumps(telemetry_data)
            return self.shm.write(data_str.encode('utf-8'))
        except Exception:
            return False

    def write_state(self, position: np.ndarray, velocity: np.ndarray, timestamp: float) -> bool:
        if not self._initialized:
            return False

        try:
            data = np.concatenate([position, velocity, [timestamp]])
            return self.state_buffer.write_numpy(data.astype(np.float64))
        except Exception:
            return False

    def read_telemetry(self) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        try:
            import json
            data = self.shm.read()
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception:
            pass
        return None

    def read_state(self) -> Optional[tuple]:
        if not self._initialized:
            return None

        try:
            data = self.state_buffer.read_numpy()
            if data is not None and len(data) >= 7:
                position = data[:3]
                velocity = data[3:6]
                timestamp = data[6]
                return position, velocity, timestamp
        except Exception:
            pass
        return None

    def close(self):
        self.shm.close()
        self.state_buffer.close()
        self._initialized = False
