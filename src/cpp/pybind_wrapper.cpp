#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ccsds_frame_sync.h"
#include "reed_solomon.h"
#include "telemetry_parser.h"
#include "shared_memory.h"

namespace py = pybind11;
using namespace aerospace;

PYBIND11_MODULE(telemetry_core, m) {
    m.doc() = "Aerospace Telemetry C++ Core Module";

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
}
