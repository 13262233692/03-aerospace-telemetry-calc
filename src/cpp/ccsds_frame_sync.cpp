#include "ccsds_frame_sync.h"
#include <cstring>
#include <algorithm>
#include <bitset>

namespace aerospace {

CCSDSFrameSync::CCSDSFrameSync()
    : sync_threshold_(2)
{
}

int CCSDSFrameSync::hamming_distance(uint32_t a, uint32_t b)
{
    uint32_t xor_val = a ^ b;
    int count = 0;
    while (xor_val) {
        count += xor_val & 1;
        xor_val >>= 1;
    }
    return count;
}

uint32_t CCSDSFrameSync::extract_uint32_msb(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + 4 > data.size()) {
        return 0;
    }
    return (static_cast<uint32_t>(data[offset]) << 24) |
           (static_cast<uint32_t>(data[offset + 1]) << 16) |
           (static_cast<uint32_t>(data[offset + 2]) << 8) |
           (static_cast<uint32_t>(data[offset + 3]));
}

bool CCSDSFrameSync::validate_frame_header(const std::vector<uint8_t>& data, size_t offset, FrameResult& result)
{
    if (offset + 10 > data.size()) {
        return false;
    }

    result.frame_version = (data[offset + 4] >> 5) & 0x07;
    result.spacecraft_id = ((static_cast<uint16_t>(data[offset + 4] & 0x1F) << 8) |
                            static_cast<uint16_t>(data[offset + 5]));
    result.virtual_channel_id = (data[offset + 6] >> 2) & 0x3F;

    uint16_t frame_length = ((static_cast<uint16_t>(data[offset + 6] & 0x03) << 8) |
                             static_cast<uint16_t>(data[offset + 7])) + 1;

    if (frame_length < MIN_FRAME_LENGTH || frame_length > MAX_FRAME_LENGTH) {
        return false;
    }

    if (offset + frame_length > data.size()) {
        return false;
    }

    result.frame_length = frame_length;

    size_t payload_start = offset + 10;
    size_t payload_end = offset + frame_length;
    if (payload_end > data.size()) {
        return false;
    }

    result.payload.assign(data.begin() + payload_start, data.begin() + payload_end);
    return true;
}

std::vector<CCSDSFrameSync::FrameResult> CCSDSFrameSync::sync_frames(const std::vector<uint8_t>& data_stream)
{
    std::vector<FrameResult> results;

    if (data_stream.size() < ASM_LENGTH + MIN_FRAME_LENGTH) {
        return results;
    }

    for (size_t i = 0; i <= data_stream.size() - ASM_LENGTH - MIN_FRAME_LENGTH; ++i) {
        uint32_t candidate = extract_uint32_msb(data_stream, i);
        int errors = hamming_distance(candidate, ASM_SYNC_WORD);

        if (errors <= sync_threshold_) {
            FrameResult result;
            result.found = false;
            result.offset = i;
            result.bit_errors = errors;

            if (validate_frame_header(data_stream, i, result)) {
                result.found = true;
                results.push_back(result);
                i += result.frame_length - 1;
            }
        }
    }

    return results;
}

std::vector<CCSDSFrameSync::FrameResult> CCSDSFrameSync::sync_frames_bitwise(const std::vector<uint8_t>& data_stream)
{
    std::vector<FrameResult> results;

    if (data_stream.size() < ASM_LENGTH + MIN_FRAME_LENGTH) {
        return results;
    }

    size_t total_bits = data_stream.size() * 8;

    for (size_t bit_offset = 0; bit_offset < total_bits - 32 - MIN_FRAME_LENGTH * 8; ++bit_offset) {
        uint32_t candidate = 0;
        for (int j = 0; j < 32; ++j) {
            size_t byte_idx = (bit_offset + j) / 8;
            size_t bit_idx = 7 - ((bit_offset + j) % 8);
            if (byte_idx < data_stream.size()) {
                candidate = (candidate << 1) | ((data_stream[byte_idx] >> bit_idx) & 1);
            }
        }

        int errors = hamming_distance(candidate, ASM_SYNC_WORD);
        if (errors <= sync_threshold_) {
            size_t byte_offset = bit_offset / 8;
            size_t bit_in_byte = bit_offset % 8;

            if (bit_in_byte == 0) {
                FrameResult result;
                result.found = false;
                result.offset = byte_offset;
                result.bit_errors = errors;

                if (validate_frame_header(data_stream, byte_offset, result)) {
                    result.found = true;
                    results.push_back(result);
                    bit_offset += result.frame_length * 8 - 1;
                }
            }
        }
    }

    return results;
}

}
