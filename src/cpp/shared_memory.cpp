#include "shared_memory.h"
#include <cstring>
#include <stdexcept>

namespace aerospace {

SharedMemoryChannel::SharedMemoryChannel()
    : capacity_(0)
    , is_open_(false)
    , is_owner_(false)
#ifdef _WIN32
    , map_handle_(NULL)
#else
    , fd_(-1)
#endif
    , mapped_ptr_(nullptr)
    , header_(nullptr)
    , data_ptr_(nullptr)
{
}

SharedMemoryChannel::~SharedMemoryChannel()
{
    close();
}

bool SharedMemoryChannel::map_memory(size_t size)
{
#ifdef _WIN32
    mapped_ptr_ = MapViewOfFile(
        map_handle_,
        FILE_MAP_ALL_ACCESS,
        0, 0,
        size
    );
    if (mapped_ptr_ == nullptr) {
        return false;
    }
#else
    mapped_ptr_ = mmap(
        nullptr,
        size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd_,
        0
    );
    if (mapped_ptr_ == MAP_FAILED) {
        mapped_ptr_ = nullptr;
        return false;
    }
#endif

    header_ = static_cast<SharedMemoryHeader*>(mapped_ptr_);
    data_ptr_ = static_cast<uint8_t*>(mapped_ptr_) + HEADER_SIZE;
    capacity_ = size - HEADER_SIZE;

    return true;
}

void SharedMemoryChannel::unmap_memory()
{
    if (mapped_ptr_) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_ptr_);
#else
        munmap(mapped_ptr_, capacity_ + HEADER_SIZE);
#endif
        mapped_ptr_ = nullptr;
        header_ = nullptr;
        data_ptr_ = nullptr;
    }
}

bool SharedMemoryChannel::create(const std::string& name, size_t capacity)
{
    if (is_open_) {
        close();
    }

    name_ = name;
    size_t total_size = capacity + HEADER_SIZE;

#ifdef _WIN32
    map_handle_ = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(total_size),
        name.c_str()
    );
    if (map_handle_ == NULL) {
        return false;
    }
#else
    std::string shm_name = "/" + name;
    fd_ = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ == -1) {
        return false;
    }
    if (ftruncate(fd_, total_size) == -1) {
        close();
        return false;
    }
#endif

    if (!map_memory(total_size)) {
        close();
        return false;
    }

    is_owner_ = true;
    is_open_ = true;

    header_->write_counter.store(0, std::memory_order_release);
    header_->read_counter.store(0, std::memory_order_release);
    header_->data_size.store(0, std::memory_order_release);
    header_->max_capacity.store(static_cast<uint32_t>(capacity), std::memory_order_release);
    header_->is_valid.store(1, std::memory_order_release);

    return true;
}

bool SharedMemoryChannel::open(const std::string& name)
{
    if (is_open_) {
        close();
    }

    name_ = name;

#ifdef _WIN32
    map_handle_ = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        name.c_str()
    );
    if (map_handle_ == NULL) {
        return false;
    }

    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(MapViewOfFile(map_handle_, FILE_MAP_READ, 0, 0, 1), &mbi, sizeof(mbi))) {
        size_t total_size = mbi.RegionSize;
        UnmapViewOfFile(mbi.BaseAddress);
        if (!map_memory(total_size)) {
            close();
            return false;
        }
    } else {
        return false;
    }
#else
    std::string shm_name = "/" + name;
    fd_ = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (fd_ == -1) {
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close();
        return false;
    }
    if (!map_memory(st.st_size)) {
        close();
        return false;
    }
#endif

    is_owner_ = false;
    is_open_ = true;

    return true;
}

void SharedMemoryChannel::close()
{
    unmap_memory();

#ifdef _WIN32
    if (map_handle_ != NULL) {
        CloseHandle(map_handle_);
        map_handle_ = NULL;
    }
#else
    if (fd_ != -1) {
        if (is_owner_) {
            std::string shm_name = "/" + name_;
            shm_unlink(shm_name.c_str());
        }
        ::close(fd_);
        fd_ = -1;
    }
#endif

    is_open_ = false;
    is_owner_ = false;
    capacity_ = 0;
}

bool SharedMemoryChannel::write(const void* data, size_t size)
{
    if (!is_open_ || !header_ || !data_ptr_) {
        return false;
    }

    if (size > capacity_) {
        return false;
    }

    std::memcpy(data_ptr_, data, size);

    header_->data_size.store(static_cast<uint32_t>(size), std::memory_order_release);
    header_->write_counter.fetch_add(1, std::memory_order_acq_rel);

    return true;
}

bool SharedMemoryChannel::read(void* buffer, size_t buffer_size, size_t& bytes_read)
{
    if (!is_open_ || !header_ || !data_ptr_) {
        return false;
    }

    uint32_t data_size = header_->data_size.load(std::memory_order_acquire);
    if (data_size > buffer_size) {
        return false;
    }

    std::memcpy(buffer, data_ptr_, data_size);
    bytes_read = data_size;

    header_->read_counter.fetch_add(1, std::memory_order_acq_rel);

    return true;
}

uint64_t SharedMemoryChannel::write_count() const
{
    if (!header_) return 0;
    return header_->write_counter.load(std::memory_order_acquire);
}

uint64_t SharedMemoryChannel::read_count() const
{
    if (!header_) return 0;
    return header_->read_counter.load(std::memory_order_acquire);
}

LockFreeRingBuffer::LockFreeRingBuffer()
    : header_(nullptr)
    , slots_(nullptr)
    , slot_count_(0)
    , slot_size_(0)
    , data_size_(0)
{
}

bool LockFreeRingBuffer::init(const std::string& shm_name, size_t slot_count, size_t slot_size)
{
    slot_count_ = slot_count;
    slot_size_ = slot_size;
    data_size_ = slot_size - 8;

    size_t total_size = sizeof(RingHeader) + slot_count * sizeof(RingSlot);

    if (!shm_.create(shm_name, total_size)) {
        return false;
    }

    uint8_t* base_ptr = static_cast<uint8_t*>(shm_.raw_data_ptr());
    header_ = reinterpret_cast<RingHeader*>(base_ptr);
    slots_ = reinterpret_cast<RingSlot*>(base_ptr + sizeof(RingHeader));

    header_->write_pos.store(0, std::memory_order_release);
    header_->read_pos.store(0, std::memory_order_release);
    header_->slot_count = static_cast<uint32_t>(slot_count);
    header_->slot_size = static_cast<uint32_t>(slot_size);

    for (size_t i = 0; i < slot_count; ++i) {
        slots_[i].sequence.store(static_cast<uint32_t>(i), std::memory_order_release);
        slots_[i].data_size = 0;
    }

    return true;
}

bool LockFreeRingBuffer::enqueue(const void* data, size_t size)
{
    if (!header_ || !slots_ || size > data_size_) {
        return false;
    }

    uint64_t pos = header_->write_pos.load(std::memory_order_acquire);
    size_t slot_idx = pos % slot_count_;
    RingSlot& slot = slots_[slot_idx];

    uint32_t expected_seq = static_cast<uint32_t>(pos);
    if (!slot.sequence.compare_exchange_strong(
            expected_seq,
            static_cast<uint32_t>(pos + 1),
            std::memory_order_acq_rel)) {
        return false;
    }

    slot.data_size = static_cast<uint32_t>(size);
    std::memcpy(slot.data, data, size);

    slot.sequence.store(static_cast<uint32_t>(pos + slot_count_), std::memory_order_release);
    header_->write_pos.store(pos + 1, std::memory_order_release);

    return true;
}

bool LockFreeRingBuffer::dequeue(void* buffer, size_t buffer_size, size_t& bytes_read)
{
    if (!header_ || !slots_) {
        return false;
    }

    uint64_t pos = header_->read_pos.load(std::memory_order_acquire);
    size_t slot_idx = pos % slot_count_;
    RingSlot& slot = slots_[slot_idx];

    uint32_t expected_seq = static_cast<uint32_t>(pos + slot_count_);
    if (!slot.sequence.compare_exchange_strong(
            expected_seq,
            static_cast<uint32_t>(pos + slot_count_ + 1),
            std::memory_order_acq_rel)) {
        return false;
    }

    size_t data_size = slot.data_size;
    if (data_size > buffer_size) {
        slot.sequence.store(static_cast<uint32_t>(pos + slot_count_), std::memory_order_release);
        return false;
    }

    std::memcpy(buffer, slot.data, data_size);
    bytes_read = data_size;

    slot.sequence.store(static_cast<uint32_t>(pos + 2 * slot_count_), std::memory_order_release);
    header_->read_pos.store(pos + 1, std::memory_order_release);

    return true;
}

size_t LockFreeRingBuffer::available_slots() const
{
    if (!header_) return 0;
    uint64_t write_pos = header_->write_pos.load(std::memory_order_acquire);
    uint64_t read_pos = header_->read_pos.load(std::memory_order_acquire);
    return slot_count_ - (write_pos - read_pos);
}

size_t LockFreeRingBuffer::used_slots() const
{
    if (!header_) return 0;
    uint64_t write_pos = header_->write_pos.load(std::memory_order_acquire);
    uint64_t read_pos = header_->read_pos.load(std::memory_order_acquire);
    return write_pos - read_pos;
}

bool LockFreeRingBuffer::is_empty() const
{
    return used_slots() == 0;
}

bool LockFreeRingBuffer::is_full() const
{
    return available_slots() == 0;
}

}
