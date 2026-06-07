#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace aerospace {

class CCSDSFrameSync {
public:
    static constexpr uint32_t ASM_SYNC_WORD = 0x1ACFFC1D;
    static constexpr size_t ASM_LENGTH = 4;
    static constexpr size_t MIN_FRAME_LENGTH = 64;
    static constexpr size_t MAX_FRAME_LENGTH = 2048;

    struct FrameResult {
        bool found;
        size_t offset;
        size_t frame_length;
        uint16_t spacecraft_id;
        uint8_t virtual_channel_id;
        uint8_t frame_version;
        std::vector<uint8_t> payload;
        int bit_errors;
    };

    CCSDSFrameSync();
    ~CCSDSFrameSync() = default;

    std::vector<FrameResult> sync_frames(const std::vector<uint8_t>& data_stream);
    std::vector<FrameResult> sync_frames_bitwise(const std::vector<uint8_t>& data_stream);

    void set_sync_threshold(int threshold) { sync_threshold_ = threshold; }
    int get_sync_threshold() const { return sync_threshold_; }

private:
    int sync_threshold_;
    std::vector<uint8_t> bit_buffer_;

    int hamming_distance(uint32_t a, uint32_t b);
    uint32_t extract_uint32_msb(const std::vector<uint8_t>& data, size_t offset);
    bool validate_frame_header(const std::vector<uint8_t>& data, size_t offset, FrameResult& result);
};

}
