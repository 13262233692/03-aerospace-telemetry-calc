#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "ccsds_frame_sync.h"
#include "reed_solomon.h"
#include "telemetry_parser.h"
#include "shared_memory.h"
#include "ekf_core.h"

namespace py = pybind11;
using namespace aerospace;

namespace {

template<typename T>
py::array_t<T> make_zero_copy_array(T* data, std::vector<ssize_t> shape,
                                     std::function<void(T*)> deleter = nullptr) {
    py::capsule free_when_done(data, [deleter, data](void* ptr) {
        if (deleter) {
            deleter(static_cast<T*>(ptr));
        }
    });

    return py::array_t<T>(shape, {}, data, free_when_done);
}

py::array_t<double> vector6_to_numpy(const Vector6& vec) {
    auto* data = new double[STATE_DIM];
    std::memcpy(data, vec.data(), sizeof(double) * STATE_DIM);

    return make_zero_copy_array<double>(data, {STATE_DIM},
        [](double* p) { delete[] p; });
}

py::array_t<double> matrix6x6_to_numpy(const Matrix6x6& mat) {
    auto* data = new double[STATE_DIM * STATE_DIM];
    for (int i = 0; i < STATE_DIM; ++i) {
        std::memcpy(data + i * STATE_DIM, mat[i].data(), sizeof(double) * STATE_DIM);
    }

    return make_zero_copy_array<double>(data, {STATE_DIM, STATE_DIM},
        [](double* p) { delete[] p; });
}

void numpy_to_vector6(const py::array_t<double>& arr, Vector6& vec) {
    if (arr.ndim() != 1 || arr.shape(0) != STATE_DIM) {
        throw std::runtime_error("Expected 1D array of size 6");
    }

    auto buf = arr.request();
    if (buf.strides[0] == sizeof(double)) {
        std::memcpy(vec.data(), buf.ptr, sizeof(double) * STATE_DIM);
    } else {
        auto ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < STATE_DIM; ++i) {
            vec[i] = ptr[i * (buf.strides[0] / sizeof(double))];
        }
    }
}

void numpy_to_matrix6x6(const py::array_t<double>& arr, Matrix6x6& mat) {
    if (arr.ndim() != 2 || arr.shape(0) != STATE_DIM || arr.shape(1) != STATE_DIM) {
        throw std::runtime_error("Expected 2D array of size 6x6");
    }

    auto buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);
    ssize_t stride0 = buf.strides[0] / sizeof(double);
    ssize_t stride1 = buf.strides[1] / sizeof(double);

    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            mat[i][j] = ptr[i * stride0 + j * stride1];
        }
    }
}

void numpy_to_vector3(const py::array_t<double>& arr, Vector3& vec) {
    if (arr.ndim() != 1 || arr.shape(0) != MEAS_DIM) {
        throw std::runtime_error("Expected 1D array of size 3");
    }

    auto buf = arr.request();
    if (buf.strides[0] == sizeof(double)) {
        std::memcpy(vec.data(), buf.ptr, sizeof(double) * MEAS_DIM);
    } else {
        auto ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < MEAS_DIM; ++i) {
            vec[i] = ptr[i * (buf.strides[0] / sizeof(double))];
        }
    }
}

void numpy_to_matrix3x3(const py::array_t<double>& arr, Matrix3x3& mat) {
    if (arr.ndim() != 2 || arr.shape(0) != MEAS_DIM || arr.shape(1) != MEAS_DIM) {
        throw std::runtime_error("Expected 2D array of size 3x3");
    }

    auto buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);
    ssize_t stride0 = buf.strides[0] / sizeof(double);
    ssize_t stride1 = buf.strides[1] / sizeof(double);

    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            mat[i][j] = ptr[i * stride0 + j * stride1];
        }
    }
}

}

PYBIND11_MODULE(telemetry_core, m) {
    m.doc() = "Aerospace Telemetry C++ Core Module";

    import_array();

    py::class_<CCSDSFrameSync>(m, "CCSDSFrameSync")
        .def(py::init<>())
        .def("sync_frames", &CCSDSFrameSync::sync_frames)
        .def("sync_frames_bitwise", &CCSDSFrameSync::sync_frames_bitwise)
        .def_property("sync_threshold",
                      &CCSDSFrameSync::get_sync_threshold,
                      &CCSDSFrameSync::set_sync_threshold)
        .def_readonly_static("ASM_SYNC_WORD", &CCSDSFrameSync::ASM_SYNC_WORD)
        .def_readonly_static("ASM_LENGTH", &CCSDSFrameSync::ASM_LENGTH)
        .def_readonly_static("MIN_FRAME_LENGTH", &CCSDSFrameSync::MIN_FRAME_LENGTH)
        .def_readonly_static("MAX_FRAME_LENGTH", &CCSDSFrameSync::MAX_FRAME_LENGTH);

    py::class_<CCSDSFrameSync::FrameResult>(m, "FrameResult")
        .def(py::init<>())
        .def_readwrite("found", &CCSDSFrameSync::FrameResult::found)
        .def_readwrite("offset", &CCSDSFrameSync::FrameResult::offset)
        .def_readwrite("frame_length", &CCSDSFrameSync::FrameResult::frame_length)
        .def_readwrite("spacecraft_id", &CCSDSFrameSync::FrameResult::spacecraft_id)
        .def_readwrite("virtual_channel_id", &CCSDSFrameSync::FrameResult::virtual_channel_id)
        .def_readwrite("frame_version", &CCSDSFrameSync::FrameResult::frame_version)
        .def_readwrite("payload", &CCSDSFrameSync::FrameResult::payload)
        .def_readwrite("bit_errors", &CCSDSFrameSync::FrameResult::bit_errors);

    py::class_<ReedSolomon>(m, "ReedSolomon")
        .def(py::init<>())
        .def("encode", &ReedSolomon::encode)
        .def("decode", [](ReedSolomon& self, const std::vector<uint8_t>& data) {
            int corrected = 0;
            auto result = self.decode(data, corrected);
            return py::make_tuple(result, corrected);
        })
        .def("encode_burst", &ReedSolomon::encode_burst)
        .def("decode_burst", [](ReedSolomon& self, const std::vector<std::vector<uint8_t>>& blocks) {
            std::vector<int> corrected;
            auto result = self.decode_burst(blocks, corrected);
            return py::make_tuple(result, corrected);
        })
        .def_property_readonly("max_correctable_errors", &ReedSolomon::get_max_correctable_errors)
        .def_readonly_static("RS_BLOCK_LENGTH", &ReedSolomon::RS_BLOCK_LENGTH)
        .def_readonly_static("RS_DATA_LENGTH", &ReedSolomon::RS_DATA_LENGTH)
        .def_readonly_static("RS_ECC_LENGTH", &ReedSolomon::RS_ECC_LENGTH);

    py::class_<AttitudeQuaternion>(m, "AttitudeQuaternion")
        .def(py::init<>())
        .def_readwrite("w", &AttitudeQuaternion::w)
        .def_readwrite("x", &AttitudeQuaternion::x)
        .def_readwrite("y", &AttitudeQuaternion::y)
        .def_readwrite("z", &AttitudeQuaternion::z)
        .def_readwrite("timestamp", &AttitudeQuaternion::timestamp);

    py::class_<GPSPseudorange>(m, "GPSPseudorange")
        .def(py::init<>())
        .def_readwrite("prn", &GPSPseudorange::prn)
        .def_readwrite("pseudorange", &GPSPseudorange::pseudorange)
        .def_readwrite("carrier_phase", &GPSPseudorange::carrier_phase)
        .def_readwrite("doppler", &GPSPseudorange::doppler)
        .def_readwrite("timestamp", &GPSPseudorange::timestamp);

    py::class_<TelemetryData>(m, "TelemetryData")
        .def(py::init<>())
        .def_readwrite("timestamp", &TelemetryData::timestamp)
        .def_readwrite("attitude", &TelemetryData::attitude)
        .def_readwrite("gps_measurements", &TelemetryData::gps_measurements)
        .def_readwrite("housekeeping", &TelemetryData::housekeeping)
        .def_readwrite("is_valid", &TelemetryData::is_valid);

    py::class_<TelemetryParser>(m, "TelemetryParser")
        .def(py::init<>())
        .def("parse_ccsds_payload", &TelemetryParser::parse_ccsds_payload)
        .def("parse_multiple_frames", &TelemetryParser::parse_multiple_frames)
        .def("parse_attitude", &TelemetryParser::parse_attitude)
        .def("parse_gps_measurement", &TelemetryParser::parse_gps_measurement)
        .def_property("attitude_format",
                      &TelemetryParser::get_attitude_format,
                      &TelemetryParser::set_attitude_format);

    py::class_<SharedMemoryChannel>(m, "SharedMemoryChannel")
        .def(py::init<>())
        .def("create", &SharedMemoryChannel::create)
        .def("open", &SharedMemoryChannel::open)
        .def("close", &SharedMemoryChannel::close)
        .def("write", [](SharedMemoryChannel& self, const py::bytes& data) {
            std::string s = data;
            return self.write(s.data(), s.size());
        })
        .def("read", [](SharedMemoryChannel& self, size_t buffer_size) {
            std::vector<uint8_t> buffer(buffer_size);
            size_t bytes_read = 0;
            bool success = self.read(buffer.data(), buffer.size(), bytes_read);
            if (!success) {
                return py::bytes();
            }
            return py::bytes(reinterpret_cast<const char*>(buffer.data()), bytes_read);
        })
        .def_property_readonly("is_open", &SharedMemoryChannel::is_open)
        .def_property_readonly("capacity", &SharedMemoryChannel::capacity)
        .def_property_readonly("write_count", &SharedMemoryChannel::write_count)
        .def_property_readonly("read_count", &SharedMemoryChannel::read_count)
        .def_readonly_static("DEFAULT_CAPACITY", &SharedMemoryChannel::DEFAULT_CAPACITY)
        .def_readonly_static("HEADER_SIZE", &SharedMemoryChannel::HEADER_SIZE);

    py::class_<LockFreeRingBuffer>(m, "LockFreeRingBuffer")
        .def(py::init<>())
        .def("init", &LockFreeRingBuffer::init)
        .def("enqueue", [](LockFreeRingBuffer& self, const py::bytes& data) {
            std::string s = data;
            return self.enqueue(s.data(), s.size());
        })
        .def("dequeue", [](LockFreeRingBuffer& self, size_t buffer_size) {
            std::vector<uint8_t> buffer(buffer_size);
            size_t bytes_read = 0;
            bool success = self.dequeue(buffer.data(), buffer.size(), bytes_read);
            if (!success) {
                return py::bytes();
            }
            return py::bytes(reinterpret_cast<const char*>(buffer.data()), bytes_read);
        })
        .def_property_readonly("available_slots", &LockFreeRingBuffer::available_slots)
        .def_property_readonly("used_slots", &LockFreeRingBuffer::used_slots)
        .def_property_readonly("is_empty", &LockFreeRingBuffer::is_empty)
        .def_property_readonly("is_full", &LockFreeRingBuffer::is_full)
        .def_readonly_static("DEFAULT_RING_SIZE", &LockFreeRingBuffer::DEFAULT_RING_SIZE);

    py::class_<ExtendedKalmanFilterCore, std::shared_ptr<ExtendedKalmanFilterCore>>(m, "ExtendedKalmanFilterCore")
        .def(py::init<>())
        .def("reset", &ExtendedKalmanFilterCore::reset)
        .def("set_process_noise", [](ExtendedKalmanFilterCore& self, py::array_t<double> Q_np) {
            Matrix6x6 Q;
            numpy_to_matrix6x6(Q_np, Q);
            self.set_process_noise(Q);
        }, py::arg("Q").noconvert())
        .def("set_state", [](ExtendedKalmanFilterCore& self, py::array_t<double> x_np,
                             py::array_t<double> P_np, double timestamp) {
            Vector6 x;
            Matrix6x6 P;
            numpy_to_vector6(x_np, x);
            numpy_to_matrix6x6(P_np, P);
            self.set_state(x, P, timestamp);
        }, py::arg("x").noconvert(), py::arg("P").noconvert(), py::arg("timestamp"))
        .def("predict", [](ExtendedKalmanFilterCore& self, py::array_t<double> F_np,
                           double dt, double timestamp) {
            Matrix6x6 F;
            numpy_to_matrix6x6(F_np, F);
            self.predict(F, dt, timestamp);
        }, py::arg("F").noconvert(), py::arg("dt"), py::arg("timestamp"))
        .def("predict_state", [](ExtendedKalmanFilterCore& self, py::array_t<double> x_new_np,
                                 py::array_t<double> F_np, double dt, double timestamp) {
            Vector6 x_new;
            Matrix6x6 F;
            numpy_to_vector6(x_new_np, x_new);
            numpy_to_matrix6x6(F_np, F);
            self.predict_state(x_new, F, dt, timestamp);
        }, py::arg("x_new").noconvert(), py::arg("F").noconvert(),
           py::arg("dt"), py::arg("timestamp"))
        .def("update_position", [](ExtendedKalmanFilterCore& self,
                                   py::array_t<double> measurement_np,
                                   py::array_t<double> R_np) {
            Vector3 measurement;
            Matrix3x3 R;
            numpy_to_vector3(measurement_np, measurement);
            numpy_to_matrix3x3(R_np, R);
            return self.update_position(measurement, R);
        }, py::arg("measurement").noconvert(), py::arg("R").noconvert())
        .def("get_state_vector", [](const ExtendedKalmanFilterCore& self) {
            return vector6_to_numpy(self.get_state().x);
        })
        .def("get_covariance_matrix", [](const ExtendedKalmanFilterCore& self) {
            return matrix6x6_to_numpy(self.get_state().P);
        })
        .def("get_timestamp", &ExtendedKalmanFilterCore::get_timestamp)
        .def_readonly_static("STATE_DIM", &STATE_DIM)
        .def_readonly_static("MEAS_DIM", &MEAS_DIM);
}
