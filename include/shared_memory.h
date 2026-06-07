#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace aerospace {

struct SharedMemoryHeader {
    std::atomic<uint64_t> write_counter;
    std::atomic<uint64_t> read_counter;
    std::atomic<uint32_t> data_size;
    std::atomic<uint32_t> max_capacity;
    std::atomic<uint8_t> is_valid;
    uint8_t padding[44];
};

static_assert(sizeof(SharedMemoryHeader) == 64, "Header must be 64 bytes for cache alignment");

class SharedMemoryChannel {
public:
    static constexpr size_t DEFAULT_CAPACITY = 1024 * 1024;
    static constexpr size_t HEADER_SIZE = sizeof(SharedMemoryHeader);

    SharedMemoryChannel();
    ~SharedMemoryChannel();

    bool create(const std::string& name, size_t capacity = DEFAULT_CAPACITY);
    bool open(const std::string& name);
    void close();

    bool write(const void* data, size_t size);
    bool read(void* buffer, size_t buffer_size, size_t& bytes_read);

    bool is_open() const { return is_open_; }
    size_t capacity() const { return capacity_; }
    uint64_t write_count() const;
    uint64_t read_count() const;

    void* raw_data_ptr() const { return data_ptr_; }
    size_t data_offset() const { return HEADER_SIZE; }

private:
    std::string name_;
    size_t capacity_;
    bool is_open_;
    bool is_owner_;

#ifdef _WIN32
    HANDLE map_handle_;
#else
    int fd_;
#endif

    void* mapped_ptr_;
    SharedMemoryHeader* header_;
    void* data_ptr_;

    bool map_memory(size_t size);
    void unmap_memory();
};

class LockFreeRingBuffer {
public:
    static constexpr size_t DEFAULT_RING_SIZE = 4096;

    LockFreeRingBuffer();
    ~LockFreeRingBuffer() = default;

    bool init(const std::string& shm_name, size_t slot_count = DEFAULT_RING_SIZE, size_t slot_size = 256);

    bool enqueue(const void* data, size_t size);
    bool dequeue(void* buffer, size_t buffer_size, size_t& bytes_read);

    size_t available_slots() const;
    size_t used_slots() const;
    bool is_empty() const;
    bool is_full() const;

private:
    struct alignas(64) RingSlot {
        std::atomic<uint32_t> sequence;
        uint32_t data_size;
        uint8_t data[256 - 8];
    };

    struct alignas(64) RingHeader {
        std::atomic<uint64_t> write_pos;
        std::atomic<uint64_t> read_pos;
        uint32_t slot_count;
        uint32_t slot_size;
    };

    SharedMemoryChannel shm_;
    RingHeader* header_;
    RingSlot* slots_;
    size_t slot_count_;
    size_t slot_size_;
    size_t data_size_;
};

}
