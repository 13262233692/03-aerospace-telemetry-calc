#include "telemetry_parser.h"
#include <cstring>
#include <stdexcept>

namespace aerospace {

TelemetryParser::TelemetryParser()
    : attitude_format_("double_be")
{
}

double TelemetryParser::parse_double_be(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 8 > data.size()) {
        throw std::out_of_range("Offset out of range for double parsing");
    }
    uint64_t raw = parse_uint64_be(data, offset);
    double result;
    std::memcpy(&result, &raw, 8);
    return result;
}

float TelemetryParser::parse_float_be(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 4 > data.size()) {
        throw std::out_of_range("Offset out of range for float parsing");
    }
    uint32_t raw = parse_uint32_be(data, offset);
    float result;
    std::memcpy(&result, &raw, 4);
    return result;
}

uint32_t TelemetryParser::parse_uint32_be(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 4 > data.size()) {
        throw std::out_of_range("Offset out of range for uint32 parsing");
    }
    return (static_cast<uint32_t>(data[offset]) << 24) |
           (static_cast<uint32_t>(data[offset + 1]) << 16) |
           (static_cast<uint32_t>(data[offset + 2]) << 8) |
           (static_cast<uint32_t>(data[offset + 3]));
}

uint16_t TelemetryParser::parse_uint16_be(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 2 > data.size()) {
        throw std::out_of_range("Offset out of range for uint16 parsing");
    }
    return (static_cast<uint16_t>(data[offset]) << 8) |
           (static_cast<uint16_t>(data[offset + 1]));
}

uint64_t TelemetryParser::parse_uint64_be(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 8 > data.size()) {
        throw std::out_of_range("Offset out of range for uint64 parsing");
    }
    return (static_cast<uint64_t>(data[offset]) << 56) |
           (static_cast<uint64_t>(data[offset + 1]) << 48) |
           (static_cast<uint64_t>(data[offset + 2]) << 40) |
           (static_cast<uint64_t>(data[offset + 3]) << 32) |
           (static_cast<uint64_t>(data[offset + 4]) << 24) |
           (static_cast<uint64_t>(data[offset + 5]) << 16) |
           (static_cast<uint64_t>(data[offset + 6]) << 8) |
           (static_cast<uint64_t>(data[offset + 7]));
}

AttitudeQuaternion TelemetryParser::parse_attitude(const std::vector<uint8_t>& data, size_t offset)
{
    AttitudeQuaternion q;
    q.timestamp = parse_uint64_be(data, offset);
    q.w = parse_double_be(data, offset + 8);
    q.x = parse_double_be(data, offset + 16);
    q.y = parse_double_be(data, offset + 24);
    q.z = parse_double_be(data, offset + 32);
    return q;
}

GPSPseudorange TelemetryParser::parse_gps_measurement(const std::vector<uint8_t>& data, size_t offset)
{
    GPSPseudorange gps;
    gps.prn = parse_uint32_be(data, offset);
    gps.pseudorange = parse_double_be(data, offset + 4);
    gps.carrier_phase = parse_double_be(data, offset + 12);
    gps.doppler = parse_double_be(data, offset + 20);
    gps.timestamp = parse_uint64_be(data, offset + 28);
    return gps;
}

TelemetryData TelemetryParser::parse_ccsds_payload(const std::vector<uint8_t>& payload)
{
    TelemetryData data;
    data.is_valid = false;

    if (payload.size() < 40) {
        return data;
    }

    try {
        size_t offset = 0;
        data.timestamp = parse_uint64_be(payload, offset);
        offset += 8;

        data.attitude = parse_attitude(payload, offset);
        offset += 40;

        uint8_t gps_count = payload[offset];
        offset += 1;

        data.gps_measurements.reserve(gps_count);
        for (int i = 0; i < gps_count && offset + 36 <= payload.size(); ++i) {
            data.gps_measurements.push_back(parse_gps_measurement(payload, offset));
            offset += 36;
        }

        if (offset + 2 <= payload.size()) {
            uint8_t hk_count = payload[offset];
            offset += 1;
            data.housekeeping.reserve(hk_count);
            for (int i = 0; i < hk_count && offset + 8 <= payload.size(); ++i) {
                data.housekeeping.push_back(parse_double_be(payload, offset));
                offset += 8;
            }
        }

        data.is_valid = true;
    } catch (const std::exception&) {
        data.is_valid = false;
    }

    return data;
}

std::vector<TelemetryData> TelemetryParser::parse_multiple_frames(const std::vector<std::vector<uint8_t>>& payloads)
{
    std::vector<TelemetryData> results;
    results.reserve(payloads.size());
    for (const auto& payload : payloads) {
        results.push_back(parse_ccsds_payload(payload));
    }
    return results;
}

}
