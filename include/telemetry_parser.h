#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace aerospace {

struct AttitudeQuaternion {
    double w, x, y, z;
    uint64_t timestamp;
};

struct GPSPseudorange {
    uint32_t prn;
    double pseudorange;
    double carrier_phase;
    double doppler;
    uint64_t timestamp;
};

struct TelemetryData {
    uint64_t timestamp;
    AttitudeQuaternion attitude;
    std::vector<GPSPseudorange> gps_measurements;
    std::vector<double> housekeeping;
    bool is_valid;
};

class TelemetryParser {
public:
    TelemetryParser();
    ~TelemetryParser() = default;

    TelemetryData parse_ccsds_payload(const std::vector<uint8_t>& payload);
    std::vector<TelemetryData> parse_multiple_frames(const std::vector<std::vector<uint8_t>>& payloads);

    AttitudeQuaternion parse_attitude(const std::vector<uint8_t>& data, size_t offset);
    GPSPseudorange parse_gps_measurement(const std::vector<uint8_t>& data, size_t offset);

    void set_attitude_format(const std::string& format) { attitude_format_ = format; }
    std::string get_attitude_format() const { return attitude_format_; }

private:
    std::string attitude_format_;
    double parse_double_be(const std::vector<uint8_t>& data, size_t offset);
    float parse_float_be(const std::vector<uint8_t>& data, size_t offset);
    uint32_t parse_uint32_be(const std::vector<uint8_t>& data, size_t offset);
    uint16_t parse_uint16_be(const std::vector<uint8_t>& data, size_t offset);
    uint64_t parse_uint64_be(const std::vector<uint8_t>& data, size_t offset);
};

}
