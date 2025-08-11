

#include <pybind11/gil.h>  // For GIL management
#include <pybind11/stl.h>
#include <numa.h>

#include "pybind_client.h"

#include <cstdlib>  // for atexit

#include "integration_utils.h"

namespace py = pybind11;

namespace mooncake {

// Python-specific wrapper functions that handle GIL and return pybind11 types
class MooncakeStorePyWrapper {
   public:
    std::shared_ptr<PyClient> store_{nullptr};

    MooncakeStorePyWrapper() : store_(PyClient::create()) {}

    pybind11::bytes get(const std::string &key) {
        if (!store_ || !store_->client_) {
            LOG(ERROR) << "Client is not initialized";
            return pybind11::bytes("\\0", 0);
        }

        const auto kNullString = pybind11::bytes("\\0", 0);

        {
            py::gil_scoped_release release_gil;
            auto buffer_handle = store_->get_buffer(key);
            if (!buffer_handle) {
                py::gil_scoped_acquire acquire_gil;
                return kNullString;
            }

void ResourceTracker::exitHandler() { getInstance().cleanupAllResources(); }

static bool isPortAvailable(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    bool available = (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0);
    close(sock);
    return available;
}

// Get a random available port between min_port and max_port
static int getRandomAvailablePort(int min_port = 12300, int max_port = 14300) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_port, max_port);

    for (int attempts = 0; attempts < 10; attempts++) {
        int port = dis(gen);
        if (isPortAvailable(port)) {
            return port;
        }
    }
    return -1;  // Failed to find available port
}

DistributedObjectStore::DistributedObjectStore() {
    // Register this instance with the global tracker
    easylog::set_min_severity(easylog::Severity::WARN);
    ResourceTracker::getInstance().registerInstance(this);
}

DistributedObjectStore::~DistributedObjectStore() {
    // Unregister from the tracker before cleanup
    ResourceTracker::getInstance().unregisterInstance(this);
}

tl::expected<void, ErrorCode> DistributedObjectStore::setup_internal(
    const std::string &local_hostname, const std::string &metadata_server,
    size_t global_segment_size, size_t local_buffer_size,
    const std::string &protocol, const std::string &rdma_devices,
    const std::string &master_server_addr) {
    this->protocol = protocol;

    // Remove port if hostname already contains one
    std::string hostname = local_hostname;
    size_t colon_pos = hostname.find(":");
    if (colon_pos == std::string::npos) {
        // Get a random available port
        int port = getRandomAvailablePort();
        if (port < 0) {
            LOG(ERROR) << "Failed to find available port";
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        // Combine hostname with port
        this->local_hostname = hostname + ":" + std::to_string(port);
    } else {
        this->local_hostname = local_hostname;
    }
    LOG(ERROR) << "setup_internal local_hostname:" << this->local_hostname;
    void **args = (protocol == "rdma") ? rdma_args(rdma_devices) : nullptr;
    auto client_opt =
        mooncake::Client::Create(this->local_hostname, metadata_server,
                                 protocol, args, master_server_addr);
    if (!client_opt) {
        LOG(ERROR) << "Failed to create client";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    client_ = *client_opt;
    // If global_segment_size is 0, skip mount segment;
    // If global_segment_size is larger than max_mr_size, split to multiple
    // segments.
    auto max_mr_size = globalConfig().max_mr_size;     // Max segment size
    uint64_t total_glbseg_size = global_segment_size;  // For logging
    uint64_t current_glbseg_size = 0;                  // For logging
    while (global_segment_size > 0) {
        size_t segment_size = std::min(global_segment_size, max_mr_size);
        global_segment_size -= segment_size;
        current_glbseg_size += segment_size;
        LOG(INFO) << "Mounting segment: " << segment_size << " bytes, "
                  << current_glbseg_size << " of " << total_glbseg_size;
        void *ptr = allocate_buffer_allocator_memory(segment_size);
        if (!ptr) {
            LOG(ERROR) << "Failed to allocate segment memory";
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        segment_ptrs_.emplace_back(ptr);
        auto mount_result = client_->MountSegment(ptr, segment_size);
        if (!mount_result.has_value()) {
            LOG(ERROR) << "Failed to mount segment: "
                       << toString(mount_result.error());
            return tl::unexpected(mount_result.error());
        }
    }
    if (total_glbseg_size == 0) {
        LOG(INFO) << "Global segment size is 0, skip mounting segment";
    }

    return {};
}

int DistributedObjectStore::setup(const std::string &local_hostname,
                                  const std::string &metadata_server,
                                  size_t global_segment_size,
                                  size_t local_buffer_size,
                                  const std::string &protocol,
                                  const std::string &rdma_devices,
                                  const std::string &master_server_addr) {
    return to_py_ret(setup_internal(
        local_hostname, metadata_server, global_segment_size, local_buffer_size,
        protocol, rdma_devices, master_server_addr));
}

tl::expected<void, ErrorCode> DistributedObjectStore::initAll_internal(
    const std::string &protocol_, const std::string &device_name,
    size_t mount_segment_size) {
    if (client_) {
        LOG(ERROR) << "Client is already initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    uint64_t buffer_allocator_size = 1024 * 1024 * 1024;
    return setup_internal("localhost:12345", "127.0.0.1:2379",
                          mount_segment_size, buffer_allocator_size, protocol_,
                          device_name);
}

int DistributedObjectStore::initAll(const std::string &protocol_,
                                    const std::string &device_name,
                                    size_t mount_segment_size) {
    return to_py_ret(
        initAll_internal(protocol_, device_name, mount_segment_size));
}

tl::expected<void, ErrorCode> DistributedObjectStore::tearDownAll_internal() {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    // Reset all resources
    client_.reset();
    client_buffer_allocator_.reset();
    segment_ptrs_.clear();
    local_hostname = "";
    device_name = "";
    protocol = "";
    return {};
}

int DistributedObjectStore::tearDownAll() {
    return to_py_ret(tearDownAll_internal());
}

tl::expected<void, ErrorCode> DistributedObjectStore::put_internal(
    const std::string &key, std::span<const char> value,
    const ReplicateConfig &config) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto alloc_result = client_buffer_allocator_->allocate(value.size_bytes());
    if (!alloc_result) {
        LOG(ERROR) << "Failed to allocate buffer for put operation, key: "
                   << key << ", value size: " << value.size();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto &buffer_handle = *alloc_result;
    memcpy(buffer_handle.ptr(), value.data(), value.size_bytes());

    std::vector<Slice> slices = split_into_slices(buffer_handle);

    auto put_result = client_->Put(key, slices, config);
    if (!put_result) {
        LOG(ERROR) << "Put operation failed with error: "
                   << toString(put_result.error());
        return tl::unexpected(put_result.error());
    }

    return {};
}

int DistributedObjectStore::put(const std::string &key,
                                std::span<const char> value,
                                const ReplicateConfig &config) {
    return to_py_ret(put_internal(key, value, config));
}

tl::expected<void, ErrorCode> DistributedObjectStore::put_batch_internal(
    const std::vector<std::string> &keys,
    const std::vector<std::span<const char>> &values,
    const ReplicateConfig &config) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (keys.size() != values.size()) {
        LOG(ERROR) << "Key and value size mismatch";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<BufferHandle> buffer_handles;
    std::unordered_map<std::string, std::vector<Slice>> batched_slices;
    batched_slices.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        auto &key = keys[i];
        auto &value = values[i];
        auto alloc_result =
            client_buffer_allocator_->allocate(value.size_bytes());
        if (!alloc_result) {
            LOG(ERROR)
                << "Failed to allocate buffer for put_batch operation, key: "
                << key << ", value size: " << value.size();
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        auto &buffer_handle = *alloc_result;
        memcpy(buffer_handle.ptr(), value.data(), value.size_bytes());
        auto slices = split_into_slices(buffer_handle);
        buffer_handles.emplace_back(std::move(*alloc_result));
        batched_slices.emplace(key, std::move(slices));
    }

    // Convert unordered_map to vector format expected by BatchPut
    std::vector<std::vector<mooncake::Slice>> ordered_batched_slices;
    ordered_batched_slices.reserve(keys.size());
    for (const auto &key : keys) {
        auto it = batched_slices.find(key);
        if (it != batched_slices.end()) {
            ordered_batched_slices.emplace_back(it->second);
        } else {
            LOG(ERROR) << "Missing slices for key: " << key;
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
    }

    auto results = client_->BatchPut(keys, ordered_batched_slices, config);

    // Check if any operations failed
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            LOG(ERROR) << "BatchPut operation failed for key '" << keys[i]
                       << "' with error: " << toString(results[i].error());
            return tl::unexpected(results[i].error());
        }
    }
    return {};
}

int DistributedObjectStore::put_batch(
    const std::vector<std::string> &keys,
    const std::vector<std::span<const char>> &values,
    const ReplicateConfig &config) {
    return to_py_ret(put_batch_internal(keys, values, config));
}

tl::expected<void, ErrorCode> DistributedObjectStore::put_parts_internal(
    const std::string &key, std::vector<std::span<const char>> values,
    const ReplicateConfig &config) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    // Calculate total size needed
    size_t total_size = 0;
    for (const auto &value : values) {
        total_size += value.size_bytes();
    }

    if (total_size == 0) {
        LOG(WARNING) << "Attempting to put empty data for key: " << key;
        return {};
    }

    // Allocate buffer using the new allocator
    auto alloc_result = client_buffer_allocator_->allocate(total_size);
    if (!alloc_result) {
        LOG(ERROR) << "Failed to allocate buffer for put_parts operation, key: "
                   << key << ", total size: " << total_size;
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    auto &buffer_handle = *alloc_result;

    // Copy all parts into the contiguous buffer
    size_t offset = 0;
    for (const auto &value : values) {
        memcpy(static_cast<char *>(buffer_handle.ptr()) + offset, value.data(),
               value.size_bytes());
        offset += value.size_bytes();
    }

    // Split into slices
    std::vector<Slice> slices = split_into_slices(buffer_handle);

    // Perform the put operation - buffer_handle will be automatically released
    auto put_result = client_->Put(key, slices, config);
    if (!put_result) {
        LOG(ERROR) << "Put operation failed with error: "
                   << toString(put_result.error());
        return tl::unexpected(put_result.error());
    }

    return {};
}

int DistributedObjectStore::put_parts(const std::string &key,
                                      std::vector<std::span<const char>> values,
                                      const ReplicateConfig &config) {
    return to_py_ret(put_parts_internal(key, values, config));
}

pybind11::bytes DistributedObjectStore::get(const std::string &key) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return pybind11::bytes("\0", 0);
    }

    const auto kNullString = pybind11::bytes("\0", 0);

    {
        py::gil_scoped_release release_gil;

        auto query_result = client_->Query(key);
        if (!query_result) {
            py::gil_scoped_acquire acquire_gil;
            return pybind11::bytes((char *)buffer_handle->ptr(),
                                   buffer_handle->size());
        }
    }

    std::vector<pybind11::bytes> get_batch(
        const std::vector<std::string> &keys) {
        const auto kNullString = pybind11::bytes("\\0", 0);
        if (!store_ || !store_->client_) {
            LOG(ERROR) << "Client is not initialized";
            py::gil_scoped_acquire acquire_gil;
            return {kNullString};
        }

        {
            py::gil_scoped_release release_gil;
            auto batch_data = store_->batch_get_buffer(keys);
            if (batch_data.empty()) {
                py::gil_scoped_acquire acquire_gil;
                return {kNullString};
            }

            py::gil_scoped_acquire acquire_gil;
            std::vector<pybind11::bytes> results;
            results.reserve(batch_data.size());

            for (const auto &data : batch_data) {
                results.emplace_back(
                    data ? pybind11::bytes((char *)data->ptr(), data->size())
                         : kNullString);
            }

            return results;
        }
    }

    pybind11::object get_tensor(const std::string &key) {
        if (!store_ || !store_->client_) {
            LOG(ERROR) << "Client is not initialized";
            return pybind11::none();
        }

        try {
            // Section with GIL released
            py::gil_scoped_release release_gil;
            auto buffer_handle = store_->get_buffer(key);
            if (!buffer_handle) {
                py::gil_scoped_acquire acquire_gil;
                return pybind11::none();
            }
            // Create contiguous buffer and copy data
            auto total_length = buffer_handle->size();
            char *exported_data = new char[total_length];
            if (!exported_data) {
                py::gil_scoped_acquire acquire_gil;
                LOG(ERROR) << "Invalid data format: insufficient data for "
                              "metadata";
                return pybind11::none();
            }
            TensorMetadata metadata;
            // Copy data from buffer to contiguous memory
            memcpy(exported_data, buffer_handle->ptr(), total_length);
            memcpy(&metadata, exported_data, sizeof(TensorMetadata));

            if (metadata.ndim < 0 || metadata.ndim > 4) {
                delete[] exported_data;
                py::gil_scoped_acquire acquire_gil;
                LOG(ERROR) << "Invalid tensor metadata: ndim=" << metadata.ndim;
                return pybind11::none();
            }

            TensorDtype dtype_enum = static_cast<TensorDtype>(metadata.dtype);
            if (dtype_enum == TensorDtype::UNKNOWN) {
                delete[] exported_data;
                py::gil_scoped_acquire acquire_gil;
                LOG(ERROR) << "Unknown tensor dtype!";
                return pybind11::none();
            }

            size_t tensor_size = total_length - sizeof(TensorMetadata);
            if (tensor_size == 0) {
                delete[] exported_data;
                py::gil_scoped_acquire acquire_gil;
                LOG(ERROR) << "Invalid data format: no tensor data found";
                return pybind11::none();
            }

        auto replica_list = query_results[i].value();
        if (replica_list.empty()) {
            LOG(ERROR) << "Empty replica list for key: " << key;
            continue;
        }

        const auto &replica = replica_list[0];
        uint64_t total_size = calculate_total_size(replica);
        if (total_size == 0) {
            continue;
        }

        auto alloc_result = client_buffer_allocator_->allocate(total_size);
        if (!alloc_result) {
            LOG(ERROR) << "Failed to allocate buffer for key: " << key;
            continue;
        }

        auto buffer_handle =
            std::make_unique<BufferHandle>(std::move(*alloc_result));
        std::vector<Slice> slices;
        allocateSlices(slices, replica, *buffer_handle);

        valid_ops.emplace_back(KeyOp{.original_index = i,
                                     .key = key,
                                     .replica_list = std::move(replica_list),
                                     .buffer_handle = std::move(buffer_handle),
                                     .slices = std::move(slices)});
    }

    if (valid_ops.empty()) {
        return final_results;
    }

    // 3. Execute batch get
    std::vector<std::string> batch_keys;
    std::vector<std::vector<Replica::Descriptor>> batch_replica_lists;
    std::unordered_map<std::string, std::vector<Slice>> batch_slices;
    batch_keys.reserve(valid_ops.size());
    batch_replica_lists.reserve(valid_ops.size());

    for (auto &op : valid_ops) {
        batch_keys.push_back(op.key);
        batch_replica_lists.push_back(op.replica_list);
        batch_slices[op.key] = op.slices;
    }

    auto batch_get_results =
        client_->BatchGet(batch_keys, batch_replica_lists, batch_slices);

    // 4. Process results and create SliceBuffers
    for (size_t i = 0; i < valid_ops.size(); ++i) {
        if (batch_get_results[i]) {
            auto &op = valid_ops[i];
            final_results[op.original_index] =
                std::make_shared<SliceBuffer>(std::move(*op.buffer_handle));
        } else {
            LOG(ERROR) << "BatchGet failed for key '" << valid_ops[i].key
                       << "': " << toString(batch_get_results[i].error());
        }
    }

    return final_results;
}

// Implementation of batch_get_buffer method
std::vector<std::shared_ptr<SliceBuffer>>
DistributedObjectStore::batch_get_buffer(const std::vector<std::string> &keys) {
    return batch_get_buffer_internal(keys);
}

tl::expected<void, ErrorCode> DistributedObjectStore::register_buffer_internal(
    void *buffer, size_t size) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    return client_->RegisterLocalMemory(buffer, size, kWildcardLocation, false,
                                        true);
}

int DistributedObjectStore::register_buffer(void *buffer, size_t size) {
    return to_py_ret(register_buffer_internal(buffer, size));
}

tl::expected<void, ErrorCode>
DistributedObjectStore::unregister_buffer_internal(void *buffer) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto unregister_result = client_->unregisterLocalMemory(buffer, true);
    if (!unregister_result) {
        LOG(ERROR) << "Unregister buffer failed with error: "
                   << toString(unregister_result.error());
        return tl::unexpected(unregister_result.error());
    }
    return {};
}

int DistributedObjectStore::unregister_buffer(void *buffer) {
    return to_py_ret(unregister_buffer_internal(buffer));
}

tl::expected<int64_t, ErrorCode> DistributedObjectStore::get_into_internal(
    const std::string &key, void *buffer, size_t size) {
    // NOTE: The buffer address must be previously registered with
    // register_buffer() for zero-copy RDMA operations to work correctly
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    // Step 1: Get object info
    auto query_result = client_->Query(key);
    if (!query_result) {
        if (query_result.error() == ErrorCode::OBJECT_NOT_FOUND) {
            VLOG(1) << "Object not found for key: " << key;
            return tl::unexpected(query_result.error());
        }
        LOG(ERROR) << "Query failed for key: " << key
                   << " with error: " << toString(query_result.error());
        return tl::unexpected(query_result.error());
    }

    auto replica_list = query_result.value();

    // Calculate total size from replica list
    if (replica_list.empty()) {
        LOG(ERROR) << "Internal error: replica_list is empty";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    auto &replica = replica_list[0];
    uint64_t total_size = calculate_total_size(replica);

    // Check if user buffer is large enough
    if (size < total_size) {
        LOG(ERROR) << "User buffer too small. Required: " << total_size
                   << ", provided: " << size;
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    // Step 2: Split user buffer according to object info and create
    // slices
    std::vector<mooncake::Slice> slices;
    uint64_t offset = 0;

    if (replica.is_memory_replica() == false) {
        while (offset < total_size) {
            auto chunk_size = std::min(total_size - offset, kMaxSliceSize);
            void *chunk_ptr = static_cast<char *>(buffer) + offset;
            slices.emplace_back(Slice{chunk_ptr, chunk_size});
            offset += chunk_size;
        }
    } else {
        for (auto &handle :
             replica.get_memory_descriptor().buffer_descriptors) {
            void *chunk_ptr = static_cast<char *>(buffer) + offset;
            slices.emplace_back(Slice{chunk_ptr, handle.size_});
            offset += handle.size_;
        }
    }

    // Step 3: Read data directly into user buffer
    auto get_result = client_->Get(key, replica_list, slices);
    if (!get_result) {
        LOG(ERROR) << "Get failed for key: " << key
                   << " with error: " << toString(get_result.error());
        return tl::unexpected(get_result.error());
    }

    return static_cast<int64_t>(total_size);
}

int DistributedObjectStore::get_into(const std::string &key, void *buffer,
                                     size_t size) {
    return to_py_ret(get_into_internal(key, buffer, size));
}

std::string DistributedObjectStore::get_hostname() const {
    return local_hostname;
}

std::vector<int> DistributedObjectStore::batch_put_from(
    const std::vector<std::string> &keys, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes, const ReplicateConfig &config) {
    auto internal_results =
        batch_put_from_internal(keys, buffers, sizes, config);
    std::vector<int> results;
    results.reserve(internal_results.size());

    for (const auto &result : internal_results) {
        results.push_back(to_py_ret(result));
    }

    return results;
}

std::vector<int> DistributedObjectStore::batch_put_from_ascend(
    const std::string key, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes, const ReplicateConfig &config) {
    auto internal_results =
        batch_put_from_internal_ascend(key, buffers, sizes, config);
    std::vector<int> results;
    results.reserve(internal_results.size());

    for (const auto &result : internal_results) {
        results.push_back(to_py_ret(result));
    }

    return results;
}

std::vector<int> DistributedObjectStore::batch_get_into(
    const std::vector<std::string> &keys, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes) {
    auto internal_results = batch_get_into_internal(keys, buffers, sizes);
    std::vector<int> results;
    results.reserve(internal_results.size());

    for (const auto &result : internal_results) {
        results.push_back(to_py_ret(result));
    }

    return results;
}

std::vector<int> DistributedObjectStore::batch_get_into_ascend(
    const std::string key, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes) {
    // auto start = std::chrono::high_resolution_clock::now();

    auto internal_results = batch_get_into_internal_ascend(key, buffers, sizes);
    std::vector<int> results;
    results.reserve(internal_results.size());

    for (const auto &result : internal_results) {
        results.push_back(to_py_ret(result));
    }
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration_call =
    //         std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // LOG(INFO) << "key: " << key << ", batch_get_into_ascend: " << duration_call.count() << "us";
    return results;
}

tl::expected<void, ErrorCode> DistributedObjectStore::put_from_internal(
    const std::string &key, void *buffer, size_t size,
    const ReplicateConfig &config) {
    // NOTE: The buffer address must be previously registered with
    // register_buffer() for zero-copy RDMA operations to work correctly
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    if (size == 0) {
        LOG(WARNING) << "Attempting to put empty data for key: " << key;
        return {};
    }

    // Create slices directly from the user buffer
    std::vector<mooncake::Slice> slices;
    uint64_t offset = 0;

    while (offset < size) {
        auto chunk_size = std::min(size - offset, kMaxSliceSize);
        void *chunk_ptr = static_cast<char *>(buffer) + offset;
        slices.emplace_back(Slice{chunk_ptr, chunk_size});
        offset += chunk_size;
    }

    auto put_result = client_->Put(key, slices, config);
    if (!put_result) {
        LOG(ERROR) << "Put operation failed with error: "
                   << toString(put_result.error());
        return tl::unexpected(put_result.error());
    }

    return {};
}

int DistributedObjectStore::put_from(const std::string &key, void *buffer,
                                     size_t size,
                                     const ReplicateConfig &config) {
    return to_py_ret(put_from_internal(key, buffer, size, config));
}

std::vector<tl::expected<int64_t, ErrorCode>>
DistributedObjectStore::batch_get_into_internal(
    const std::vector<std::string> &keys, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes) {
    // Validate preconditions
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return std::vector<tl::expected<int64_t, ErrorCode>>(
            keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    if (keys.size() != buffers.size() || keys.size() != sizes.size()) {
        LOG(ERROR) << "Input vector sizes mismatch: keys=" << keys.size()
                   << ", buffers=" << buffers.size()
                   << ", sizes=" << sizes.size();
        return std::vector<tl::expected<int64_t, ErrorCode>>(
            keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    const size_t num_keys = keys.size();
    std::vector<tl::expected<int64_t, ErrorCode>> results;
    results.reserve(num_keys);

    if (num_keys == 0) {
        return results;
    }

    // Query metadata for all keys
    const auto query_results = client_->BatchQuery(keys);

    // Process each key individually and prepare for batch transfer
    struct ValidKeyInfo {
        std::string key;
        size_t original_index;
        std::vector<Replica::Descriptor> replica_list;
        std::vector<Slice> slices;
        uint64_t total_size;
    };

    std::vector<ValidKeyInfo> valid_operations;
    valid_operations.reserve(num_keys);

    for (size_t i = 0; i < num_keys; ++i) {
        const auto &key = keys[i];

        // Handle query failures
        if (!query_results[i]) {
            const auto error = query_results[i].error();
            results.emplace_back(tl::unexpected(error));
            if (error != ErrorCode::OBJECT_NOT_FOUND) {
                LOG(ERROR) << "Query failed for key '" << key
                           << "': " << toString(error);
            }
            continue;
        }

        // Validate replica list
        auto replica_list = query_results[i].value();
        if (replica_list.empty()) {
            LOG(ERROR) << "Empty replica list for key: " << key;
            results.emplace_back(tl::unexpected(ErrorCode::INVALID_REPLICA));
            continue;
        }

        // Calculate required buffer size
        const auto &replica = replica_list[0];
        uint64_t total_size = calculate_total_size(replica);

        // Validate buffer capacity
        if (sizes[i] < total_size) {
            LOG(ERROR) << "Buffer too small for key '" << key
                       << "': required=" << total_size
                       << ", available=" << sizes[i];
            results.emplace_back(tl::unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }

        // Create slices for this key's buffer
        std::vector<Slice> key_slices;
        uint64_t offset = 0;
        if (replica.is_memory_replica() == false) {
            while (offset < total_size) {
                auto chunk_size = std::min(total_size - offset, kMaxSliceSize);
                void *chunk_ptr = static_cast<char *>(buffers[i]) + offset;
                key_slices.emplace_back(Slice{chunk_ptr, chunk_size});
                offset += chunk_size;
            }
        } else {
            for (auto &handle :
                 replica.get_memory_descriptor().buffer_descriptors) {
                void *chunk_ptr = static_cast<char *>(buffers[i]) + offset;
                key_slices.emplace_back(Slice{chunk_ptr, handle.size_});
                offset += handle.size_;
            }
        }

        // Store operation info for batch processing
        valid_operations.push_back({.key = key,
                                    .original_index = i,
                                    .replica_list = std::move(replica_list),
                                    .slices = std::move(key_slices),
                                    .total_size = total_size});

        // Set success result (actual bytes transferred)
        results.emplace_back(static_cast<int64_t>(total_size));
    }

    // Early return if no valid operations
    if (valid_operations.empty()) {
        return results;
    }

    // Prepare batch transfer data structures
    std::vector<std::string> batch_keys;
    std::vector<std::vector<Replica::Descriptor>> batch_replica_lists;
    std::unordered_map<std::string, std::vector<Slice>> batch_slices;

    batch_keys.reserve(valid_operations.size());
    batch_replica_lists.reserve(valid_operations.size());

    for (const auto &op : valid_operations) {
        batch_keys.push_back(op.key);
        batch_replica_lists.push_back(op.replica_list);
        batch_slices[op.key] = op.slices;
    }

    // Execute batch transfer
    const auto batch_get_results =
        client_->BatchGet(batch_keys, batch_replica_lists, batch_slices);

    // Process transfer results
    for (size_t j = 0; j < batch_get_results.size(); ++j) {
        const auto &op = valid_operations[j];

        if (!batch_get_results[j]) {
            const auto error = batch_get_results[j].error();
            LOG(ERROR) << "BatchGet failed for key '" << op.key
                       << "': " << toString(error);
            results[op.original_index] = tl::unexpected(error);
        }
    }

    return results;
}

std::vector<tl::expected<int64_t, ErrorCode>>
DistributedObjectStore::batch_get_into_internal_ascend(
    const std::string key, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes) {
    // LOG(INFO) << "GET KEY start: " << key; 
    // Validate preconditions
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return std::vector<tl::expected<int64_t, ErrorCode>>(
            1, tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    if (buffers.size() != sizes.size()) {
        LOG(ERROR) << "Input vector sizes mismatch: keys=" << 1
                   << ", buffers=" << buffers.size()
                   << ", sizes=" << sizes.size();
        return std::vector<tl::expected<int64_t, ErrorCode>>(
            1, tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    const size_t num_keys = 1;
    std::vector<tl::expected<int64_t, ErrorCode>> results;
    results.reserve(num_keys);

    if (num_keys == 0) {
        return results;
    }
    std::vector<std::string> keys;
    keys.reserve(1);
    keys.emplace_back(key);
    // Query metadata for all keys
    const auto query_results = client_->BatchQuery(keys);

    // Process each key individually and prepare for batch transfer
    struct ValidKeyInfo {
        std::string key;
        size_t original_index;
        std::vector<Replica::Descriptor> replica_list;
        std::vector<Slice> slices;
        uint64_t total_size;
    };

    std::vector<ValidKeyInfo> valid_operations;
    valid_operations.reserve(num_keys);

    for (size_t i = 0; i < num_keys; ++i) {
        const auto &key = keys[i];

        // Handle query failures
        if (!query_results[i]) {
            const auto error = query_results[i].error();
            results.emplace_back(tl::unexpected(error));
            if (error != ErrorCode::OBJECT_NOT_FOUND) {
                LOG(ERROR) << "Query failed for key '" << key
                           << "': " << toString(error);
            }
            continue;
        }

        // Validate replica list
        auto replica_list = query_results[i].value();
        if (replica_list.empty()) {
            LOG(ERROR) << "Empty replica list for key: " << key;
            results.emplace_back(tl::unexpected(ErrorCode::INVALID_REPLICA));
            continue;
        }

        // Calculate required buffer size
        const auto &replica = replica_list[0];
        uint64_t total_size = calculate_total_size(replica);
        int total_key_size = 0;
        for (size_t k = 0; k < sizes.size(); ++k) {
            total_key_size += sizes[k];
        }
        // LOG(INFO) << "KEY: '" << key
                    // << "': required=" << total_size
                    // << ", available=" << total_key_size;
        // Validate buffer capacity
        if (total_key_size < total_size) {
            LOG(ERROR) << "Buffer too small for key '" << key
                       << "': required=" << total_size
                       << ", available=" << total_key_size;
            results.emplace_back(tl::unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }
        std::vector<Slice> key_slices;
        // Create slices for this key's buffer
        for (size_t j = 0; j < buffers.size(); ++j) {
            uint64_t offset = 0;
            if (replica.is_memory_replica() == false) {
                key_slices.emplace_back(Slice{buffers[j], sizes[j]});
            } else {
                key_slices.emplace_back(Slice{buffers[j], sizes[j]});
            }
        }

        // Store operation info for batch processing
        valid_operations.push_back({.key = key,
                                    .original_index = i,
                                    .replica_list = std::move(replica_list),
                                    .slices = std::move(key_slices),
                                    .total_size = total_size});

        // Set success result (actual bytes transferred)
        results.emplace_back(static_cast<int64_t>(total_size));
    }

    // Early return if no valid operations
    if (valid_operations.empty()) {
        return results;
    }

    // Prepare batch transfer data structures
    std::vector<std::string> batch_keys;
    std::vector<std::vector<Replica::Descriptor>> batch_replica_lists;
    std::unordered_map<std::string, std::vector<Slice>> batch_slices;

    batch_keys.reserve(valid_operations.size());
    batch_replica_lists.reserve(valid_operations.size());

    for (const auto &op : valid_operations) {
        batch_keys.push_back(op.key);
        batch_replica_lists.push_back(op.replica_list);
        batch_slices[op.key] = op.slices;
    }

    // Execute batch transfer
    const auto batch_get_results =
        client_->BatchGet(batch_keys, batch_replica_lists, batch_slices);

    // Process transfer results
    for (size_t j = 0; j < batch_get_results.size(); ++j) {
        const auto &op = valid_operations[j];

        if (!batch_get_results[j]) {
            const auto error = batch_get_results[j].error();
            LOG(ERROR) << "BatchGet failed for key '" << op.key
                       << "': " << toString(error);
            results[op.original_index] = tl::unexpected(error);
        }
    }
    // LOG(INFO) << "GET KEY end: " << key << "end"; 

    return results;
}

std::vector<tl::expected<void, ErrorCode>>
DistributedObjectStore::batch_put_from_internal(
    const std::vector<std::string> &keys, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes, const ReplicateConfig &config) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return std::vector<tl::expected<void, ErrorCode>>(
            keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    if (keys.size() != buffers.size() || keys.size() != sizes.size()) {
        LOG(ERROR) << "Mismatched sizes for keys, buffers, and sizes";
        return std::vector<tl::expected<void, ErrorCode>>(
            keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    std::unordered_map<std::string, std::vector<mooncake::Slice>> all_slices;

    // Create slices from user buffers
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string &key = keys[i];
        void *buffer = buffers[i];
        size_t size = sizes[i];

        std::vector<mooncake::Slice> slices;
        uint64_t offset = 0;

        while (offset < size) {
            auto chunk_size = std::min(size - offset, kMaxSliceSize);
            void *chunk_ptr = static_cast<char *>(buffer) + offset;
            slices.emplace_back(Slice{chunk_ptr, chunk_size});
            offset += chunk_size;
        }

        all_slices[key] = std::move(slices);
    }

    std::vector<std::vector<mooncake::Slice>> ordered_batched_slices;
    ordered_batched_slices.reserve(keys.size());
    for (const auto &key : keys) {
        auto it = all_slices.find(key);
        if (it != all_slices.end()) {
            ordered_batched_slices.emplace_back(it->second);
        } else {
            LOG(ERROR) << "Missing slices for key: " << key;
            return std::vector<tl::expected<void, ErrorCode>>(
                keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
        }
    }

    // Call client BatchPut and return the vector<expected> directly
    return client_->BatchPut(keys, ordered_batched_slices, config);
}

std::vector<tl::expected<void, ErrorCode>>
DistributedObjectStore::batch_put_from_internal_ascend(
    const std::string key, const std::vector<void *> &buffers,
    const std::vector<size_t> &sizes, const ReplicateConfig &config) {
    LOG(INFO) << "PUT KEY start: " << key; 
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return std::vector<tl::expected<void, ErrorCode>>(
            1, tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    if (buffers.size() != sizes.size()) {
        LOG(ERROR) << "Mismatched sizes for key, buffers, and sizes";
        return std::vector<tl::expected<void, ErrorCode>>(
            1, tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    // std::unordered_map<std::string, std::vector<mooncake::Slice>> all_slices;

    // Create slices from user buffers
    std::vector<mooncake::Slice> slices;
    slices.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        void *buffer = buffers[i];
        size_t size = sizes[i];
        slices.emplace_back(Slice{buffer, size});
    }
    std::vector<std::vector<mooncake::Slice>> ordered_batched_slices;
    ordered_batched_slices.reserve(1);
    ordered_batched_slices.emplace_back(slices);

    std::vector<std::string> keys;
    keys.reserve(1);
    keys.emplace_back(key);
    LOG(ERROR) << "batch put keys size:" << keys.size() << ", ordered_batched_slices size:" << ordered_batched_slices.size()
    << ", slice size len:" << slices.size();

    // Call client BatchPut and return the vector<expected> directly
    return client_->BatchPut(keys, ordered_batched_slices, config);
}

std::vector<tl::expected<bool, ErrorCode>>
DistributedObjectStore::batchIsExist_internal(
    const std::vector<std::string> &keys) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return std::vector<tl::expected<bool, ErrorCode>>(
            keys.size(), tl::unexpected(ErrorCode::INVALID_PARAMS));
    }

    if (keys.empty()) {
        LOG(WARNING) << "Empty keys vector provided to batchIsExist_internal";
        return std::vector<tl::expected<bool, ErrorCode>>();
    }

    // Call client BatchIsExist and return the vector<expected> directly
    return client_->BatchIsExist(keys);
}

int DistributedObjectStore::put_from_with_metadata(
    const std::string &key, void *buffer, void *metadata_buffer, size_t size,
    size_t metadata_size, const ReplicateConfig &config) {
    // NOTE: The buffer address must be previously registered with
    // register_buffer() for zero-copy RDMA operations to work correctly
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return -1;
    }

    if (size == 0) {
        LOG(WARNING) << "Attempting to put empty data for key: " << key;
        return 0;
    }

    // Create slices directly from the user buffer
    std::vector<mooncake::Slice> slices;
    // Add metadata slice
    uint64_t metadata_offset = 0;
    while (metadata_offset < metadata_size) {
        auto metadata_chunk_size =
            std::min(metadata_size - metadata_offset, kMaxSliceSize);
        void *metadata_chunk_ptr =
            static_cast<char *>(metadata_buffer) + metadata_offset;
        slices.emplace_back(Slice{metadata_chunk_ptr, metadata_chunk_size});
        metadata_offset += metadata_chunk_size;
    }

    uint64_t offset = 0;
    while (offset < size) {
        auto chunk_size = std::min(size - offset, kMaxSliceSize);
        void *chunk_ptr = static_cast<char *>(buffer) + offset;
        slices.emplace_back(Slice{chunk_ptr, chunk_size});
        offset += chunk_size;
    }
    auto put_result = client_->Put(key, slices, config);
    if (!put_result) {
        LOG(ERROR) << "Put operation failed with error: "
                   << toString(put_result.error());
        return -toInt(put_result.error());
    }
    return 0;
}

pybind11::object DistributedObjectStore::get_tensor(const std::string &key) {
    if (!client_) {
        LOG(ERROR) << "Client is not initialized";
        return pybind11::none();
    }

    try {
        // Query object info first
        auto query_result = client_->Query(key);
        if (!query_result) {
            py::gil_scoped_acquire acquire_gil;
            // Convert bytes to tensor using torch.from_numpy
            pybind11::object np_array;
            int dtype_index = static_cast<int>(dtype_enum);
            if (dtype_index >= 0 &&
                dtype_index < static_cast<int>(array_creators.size())) {
                np_array = array_creators[dtype_index](
                    exported_data, sizeof(TensorMetadata), tensor_size);
            } else {
                LOG(ERROR) << "Unsupported dtype enum: " << dtype_index;
                return pybind11::none();
            }

            if (metadata.ndim > 0) {
                std::vector<int> shape_vec;
                for (int i = 0; i < metadata.ndim; i++) {
                    shape_vec.push_back(metadata.shape[i]);
                }
                py::tuple shape_tuple = py::cast(shape_vec);
                np_array = np_array.attr("reshape")(shape_tuple);
            }
            pybind11::object tensor =
                torch_module().attr("from_numpy")(np_array);
            return tensor;

        } catch (const pybind11::error_already_set &e) {
            LOG(ERROR) << "Failed to get tensor data: " << e.what();
            return pybind11::none();
        }
    }

    int put_tensor(const std::string &key, pybind11::object tensor) {
        if (!store_ || !store_->client_) {
            LOG(ERROR) << "Client is not initialized";
            return -static_cast<int>(ErrorCode::INVALID_PARAMS);
        }
        try {
            if (!(tensor.attr("__class__")
                      .attr("__name__")
                      .cast<std::string>()
                      .find("Tensor") != std::string::npos)) {
                LOG(ERROR) << "Input is not a PyTorch tensor";
                return -static_cast<int>(ErrorCode::INVALID_PARAMS);
            }

            uintptr_t data_ptr = tensor.attr("data_ptr")().cast<uintptr_t>();
            size_t numel = tensor.attr("numel")().cast<size_t>();
            size_t element_size = tensor.attr("element_size")().cast<size_t>();
            size_t tensor_size = numel * element_size;

            pybind11::object shape_obj = tensor.attr("shape");
            pybind11::object dtype_obj = tensor.attr("dtype");

            TensorDtype dtype_enum = get_tensor_dtype(dtype_obj);
            if (dtype_enum == TensorDtype::UNKNOWN) {
                LOG(ERROR) << "Unsupported tensor dtype!";
                return -static_cast<int>(ErrorCode::INVALID_PARAMS);
            }

            pybind11::tuple shape_tuple =
                pybind11::cast<pybind11::tuple>(shape_obj);
            int32_t ndim = static_cast<int32_t>(shape_tuple.size());
            if (ndim > 4) {
                LOG(ERROR) << "Tensor has more than 4 dimensions: " << ndim;
                return -static_cast<int>(ErrorCode::INVALID_PARAMS);
            }

            TensorMetadata metadata;
            metadata.dtype = static_cast<int32_t>(dtype_enum);
            metadata.ndim = ndim;

            for (int i = 0; i < 4; i++) {
                if (i < ndim) {
                    metadata.shape[i] = shape_tuple[i].cast<int32_t>();
                } else {
                    metadata.shape[i] = -1;
                }
            }

            // Section with GIL released
            py::gil_scoped_release release_gil;
            char *buffer = reinterpret_cast<char *>(data_ptr);
            char *metadata_buffer = reinterpret_cast<char *>(&metadata);
            std::vector<std::span<const char>> values;
            values.emplace_back(
                std::span<const char>(metadata_buffer, sizeof(TensorMetadata)));
            values.emplace_back(std::span<const char>(buffer, tensor_size));

            // Use put_parts to put metadata and tensor together
            auto put_result = store_->put_parts_internal(key, values);
            if (!put_result) {
                return -static_cast<int>(put_result.error());
            }

            return 0;
        } catch (const pybind11::error_already_set &e) {
            LOG(ERROR) << "Failed to access tensor data: " << e.what();
            return -static_cast<int>(ErrorCode::INVALID_PARAMS);
        }
    }
};

PYBIND11_MODULE(store, m) {
    // Define the ReplicateConfig class
    py::class_<ReplicateConfig>(m, "ReplicateConfig")
        .def(py::init<>())
        .def_readwrite("replica_num", &ReplicateConfig::replica_num)
        .def_readwrite("with_soft_pin", &ReplicateConfig::with_soft_pin)
        .def_readwrite("preferred_segment", &ReplicateConfig::preferred_segment)
        .def("__str__", [](const ReplicateConfig &config) {
            std::ostringstream oss;
            oss << config;
            return oss.str();
        });

    // Define the BufferHandle class
    py::class_<BufferHandle, std::shared_ptr<BufferHandle>>(
        m, "BufferHandle", py::buffer_protocol())
        .def("ptr",
             [](const BufferHandle &self) {
                 // Return the pointer as an integer for Python
                 return reinterpret_cast<uintptr_t>(self.ptr());
             })
        .def("size", &BufferHandle::size)
        .def("__len__", &BufferHandle::size)
        .def_buffer([](BufferHandle &self) -> py::buffer_info {
            // BufferHandle now always contains contiguous memory
            if (self.size() > 0) {
                return py::buffer_info(
                    self.ptr(),   /* Pointer to buffer */
                    sizeof(char), /* Size of one scalar */
                    py::format_descriptor<
                        char>::format(),   /* Python struct-style
                                              format descriptor */
                    1,                     /* Number of dimensions */
                    {(size_t)self.size()}, /* Buffer dimensions */
                    {sizeof(char)} /* Strides (in bytes) for each index */
                );
            } else {
                // Empty buffer
                return py::buffer_info(
                    nullptr,      /* Pointer to buffer */
                    sizeof(char), /* Size of one scalar */
                    py::format_descriptor<
                        char>::format(), /* Python struct-style
                                            format descriptor */
                    1,                   /* Number of dimensions */
                    {0},                 /* Buffer dimensions */
                    {sizeof(char)}       /* Strides (in bytes) for each index */
                );
            }
        });

    // Create a wrapper that exposes DistributedObjectStore with Python-specific
    // methods
    py::class_<MooncakeStorePyWrapper>(m, "MooncakeDistributedStore")
        .def(py::init<>())
        .def("setup",
             [](MooncakeStorePyWrapper &self, const std::string &local_hostname,
                const std::string &metadata_server,
                size_t global_segment_size = 1024 * 1024 * 16,
                size_t local_buffer_size = 1024 * 1024 * 16,
                const std::string &protocol = "tcp",
                const std::string &rdma_devices = "",
                const std::string &master_server_addr = "127.0.0.1:50051") {
                 if (!self.store_) {
                     self.store_ = PyClient::create();
                 }
                 return self.store_->setup(local_hostname, metadata_server,
                                           global_segment_size,
                                           local_buffer_size, protocol,
                                           rdma_devices, master_server_addr);
             })
        .def("init_all",
             [](MooncakeStorePyWrapper &self, const std::string &protocol,
                const std::string &device_name,
                size_t mount_segment_size = 1024 * 1024 * 16) {
                 return self.store_->initAll(protocol, device_name,
                                             mount_segment_size);
             })
        .def("get", &MooncakeStorePyWrapper::get)
        .def("get_batch", &MooncakeStorePyWrapper::get_batch)
        .def(
            "get_buffer",
            [](MooncakeStorePyWrapper &self, const std::string &key) {
                py::gil_scoped_release release;
                return self.store_->get_buffer(key);
            },
            py::return_value_policy::take_ownership)
        .def(
            "batch_get_buffer",
            [](MooncakeStorePyWrapper &self,
               const std::vector<std::string> &keys) {
                py::gil_scoped_release release;
                return self.store_->batch_get_buffer(keys);
            },
            py::return_value_policy::take_ownership)
        .def("remove",
             [](MooncakeStorePyWrapper &self, const std::string &key) {
                 py::gil_scoped_release release;
                 return self.store_->remove(key);
             })
        .def(
            "remove_by_regex",
            [](MooncakeStorePyWrapper &self, const std::string &str) {
                py::gil_scoped_release release;
                return self.store_->removeByRegex(str);
            },
            py::arg("regex_pattern"),
            "Removes objects from the store whose keys match the given "
            "regular expression.")
        .def("remove_all",
             [](MooncakeStorePyWrapper &self) {
                 py::gil_scoped_release release;
                 return self.store_->removeAll();
             })
        .def("is_exist",
             [](MooncakeStorePyWrapper &self, const std::string &key) {
                 py::gil_scoped_release release;
                 return self.store_->isExist(key);
             })
        .def(
            "batch_is_exist",
            [](MooncakeStorePyWrapper &self,
               const std::vector<std::string> &keys) {
                py::gil_scoped_release release;
                return self.store_->batchIsExist(keys);
            },
            py::arg("keys"),
            "Check if multiple objects exist. Returns list of results: 1 if "
            "exists, 0 if not exists, -1 if error")
        .def("close",
             [](MooncakeStorePyWrapper &self) {
                 if (!self.store_) return 0;
                 int rc = self.store_->tearDownAll();
                 self.store_.reset();
                 return rc;
             })
        .def("get_size",
             [](MooncakeStorePyWrapper &self, const std::string &key) {
                 py::gil_scoped_release release;
                 return self.store_->getSize(key);
             })
        .def("get_tensor", &MooncakeStorePyWrapper::get_tensor, py::arg("key"),
             "Get a PyTorch tensor from the store")
        .def("put_tensor", &MooncakeStorePyWrapper::put_tensor, py::arg("key"),
             py::arg("tensor"), "Put a PyTorch tensor into the store")
        .def(
            "register_buffer",
            [](MooncakeStorePyWrapper &self, uintptr_t buffer_ptr,
               size_t size) {
                // Register memory buffer for RDMA operations
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.store_->register_buffer(buffer, size);
            },
            py::arg("buffer_ptr"), py::arg("size"),
            "Register a memory buffer for direct access operations")
        .def(
            "unregister_buffer",
            [](MooncakeStorePyWrapper &self, uintptr_t buffer_ptr) {
                // Unregister memory buffer
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.store_->unregister_buffer(buffer);
            },
            py::arg("buffer_ptr"),
            "Unregister a previously registered memory "
            "buffer for direct access operations")
        .def(
            "get_into",
            [](MooncakeStorePyWrapper &self, const std::string &key,
               uintptr_t buffer_ptr, size_t size) {
                // Get data directly into user-provided buffer
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.store_->get_into(key, buffer, size);
            },
            py::arg("key"), py::arg("buffer_ptr"), py::arg("size"),
            "Get object data directly into a pre-allocated buffer")
        .def(
            "batch_get_into",
            [](MooncakeStorePyWrapper &self,
               const std::vector<std::string> &keys,
               const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.store_->batch_get_into(keys, buffers, sizes);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            "Get object data directly into pre-allocated buffers for "
            "multiple "
            "keys")
        .def(
            "put_from",
            [](MooncakeStorePyWrapper &self, const std::string &key,
               uintptr_t buffer_ptr, size_t size,
               const ReplicateConfig &config = ReplicateConfig{}) {
                // Put data directly from user-provided buffer
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.store_->put_from(key, buffer, size, config);
            },
            py::arg("key"), py::arg("buffer_ptr"), py::arg("size"),
            py::arg("config") = ReplicateConfig{},
            "Put object data directly from a pre-allocated buffer")
        .def(
            "put_from_with_metadata",
            [](MooncakeStorePyWrapper &self, const std::string &key,
               uintptr_t buffer_ptr, uintptr_t metadata_buffer_ptr, size_t size,
               size_t metadata_size,
               const ReplicateConfig &config = ReplicateConfig{}) {
                // Put data directly from user-provided buffer with
                // metadata
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                void *metadata_buffer =
                    reinterpret_cast<void *>(metadata_buffer_ptr);
                py::gil_scoped_release release;
                return self.store_->put_from_with_metadata(
                    key, buffer, metadata_buffer, size, metadata_size, config);
            },
            py::arg("key"), py::arg("buffer_ptr"),
            py::arg("metadata_buffer_ptr"), py::arg("size"),
            py::arg("metadata_size"), py::arg("config") = ReplicateConfig{},
            "Put object data directly from a pre-allocated buffer with "
            "metadata")
        .def(
            "batch_put_from",
            [](MooncakeStorePyWrapper &self,
               const std::vector<std::string> &keys,
               const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes,
               const ReplicateConfig &config = ReplicateConfig{}) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.store_->batch_put_from(keys, buffers, sizes,
                                                   config);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("config") = ReplicateConfig{},
            "Put object data directly from pre-allocated buffers for "
            "multiple "
            "keys")
        .def(
            "put",
            [](MooncakeStorePyWrapper &self, const std::string &key,
               py::buffer buf,
               const ReplicateConfig &config = ReplicateConfig{}) {
                py::buffer_info info = buf.request(/*writable=*/false);
                py::gil_scoped_release release;
                return self.store_->put(
                    key,
                    std::span<const char>(static_cast<char *>(info.ptr),
                                          static_cast<size_t>(info.size)),
                    config);
            },
            py::arg("key"), py::arg("value"),
            py::arg("config") = ReplicateConfig{})
        .def(
            "put_parts",
            [](MooncakeStorePyWrapper &self, const std::string &key,
               py::args parts,
               const ReplicateConfig &config = ReplicateConfig{}) {
                // 1) Python buffer → span
                std::vector<py::buffer_info> infos;
                std::vector<std::span<const char>> spans;
                infos.reserve(parts.size());
                spans.reserve(parts.size());

                for (auto &obj : parts) {
                    py::buffer buf = py::reinterpret_borrow<py::buffer>(obj);
                    infos.emplace_back(buf.request(false));
                    const auto &info = infos.back();
                    if (info.ndim != 1 || info.itemsize != 1)
                        throw std::runtime_error(
                            "parts must be 1-D bytes-like");

                    spans.emplace_back(static_cast<const char *>(info.ptr),
                                       static_cast<size_t>(info.size));
                }

                // 2) Call C++ function
                py::gil_scoped_release unlock;
                return self.store_->put_parts(key, spans, config);
            },
            py::arg("key"), py::arg("config") = ReplicateConfig{})
        .def(
            "put_batch",
            [](MooncakeStorePyWrapper &self,
               const std::vector<std::string> &keys,
               const std::vector<py::buffer> &buffers,
               const ReplicateConfig &config = ReplicateConfig{}) {
                // Convert pybuffers to spans without copying
                std::vector<py::buffer_info> infos;
                std::vector<std::span<const char>> spans;
                infos.reserve(buffers.size());
                spans.reserve(buffers.size());

                for (const auto &buf : buffers) {
                    infos.emplace_back(buf.request(/*writable=*/false));
                    const auto &info = infos.back();
                    spans.emplace_back(static_cast<const char *>(info.ptr),
                                       static_cast<size_t>(info.size));
                }

                py::gil_scoped_release release;
                return self.store_->put_batch(keys, spans, config);
            },
            py::arg("keys"), py::arg("values"),
            py::arg("config") = ReplicateConfig{})
        .def("get_hostname", [](MooncakeStorePyWrapper &self) {
            return self.store_->get_hostname();
        .def(
            "batch_put_from_ascend",
            [](DistributedObjectStore &self,
               const std::string key,
               const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes,
               const ReplicateConfig &config = ReplicateConfig{}) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.batch_put_from_ascend(key, buffers, sizes, config);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("config") = ReplicateConfig{},
            "Put object data directly from pre-allocated buffers for "
            "multiple "
            "keys")
        .def(
            "batch_get_into_ascend",
            [](DistributedObjectStore &self,
               const std::string key,
               const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.batch_get_into_ascend(key, buffers, sizes);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            "Get object data directly into pre-allocated buffers for "
            "multiple "
            "keys");
        });

    // Expose NUMA binding as a module-level function (no self required)
    m.def(
        "bind_to_numa_node",
        [](int node) {
            if (numa_available() < 0) {
                LOG(WARNING)
                    << "NUMA is not available on this system; binding skipped";
                return;
            }

            int max_node = numa_max_node();
            if (node < 0 || node > max_node) {
                LOG(WARNING) << "Invalid NUMA node: " << node
                             << ". Valid range: 0-" << max_node;
            }

            if (numa_run_on_node(node) != 0) {
                LOG(WARNING) << "numa_run_on_node failed for node " << node;
            }

            // Prefer this NUMA node for future allocations but allow fallback
            numa_set_bind_policy(0);  // non-strict binding
            numa_set_preferred(node);
        },
        py::arg("node"),
        "Bind the current thread and memory allocation preference to the "
        "specified NUMA node");
}

}  // namespace mooncake
