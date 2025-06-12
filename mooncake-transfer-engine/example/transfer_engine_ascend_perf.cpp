// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/time.h>

#include <signal.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "common/base/status.h"
#include "transfer_engine.h"
#include "transport/transport.h"

#include "acl/acl.h"
#include "hccl/hccl.h"

DEFINE_string(local_server_name, "10.20.130.154:12345",
              "Local server name for segment discovery");
DEFINE_string(metadata_server, "P2PHANDSHAKE", "etcd server host address");
DEFINE_string(mode, "initiator",
              "Running mode: initiator or target. Initiator node read/write "
              "data blocks from target node");
DEFINE_string(operation, "write", "Operation type: read or write");
DEFINE_string(protocol, "hccl", "Transfer protocol: rdma|tcp|hccl");
DEFINE_string(segment_id, "10.20.130.154:12346", "Segment ID to access data");
DEFINE_int32(batch_size, 20, "Batch size");
DEFINE_uint64(block_size, 32768, "Block size for each transfer request");
DEFINE_uint64(block_iteration, 8, "number of iterations of the block");
DEFINE_bool(auto_discovery, false, "Enable auto discovery");
DEFINE_uint64(device_id, 0, "The device ID of this machine");
DEFINE_string(segment_id_1, "NA", "A segment ID that a sender wants to another receiver");
DEFINE_uint64(recv_num, 1, "Num of coonections received by the receiver");
DEFINE_uint64(send_index, 0, "which one is sent to the same receiver");
DEFINE_string(report_unit, "GB", "Report unit: GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb");
DEFINE_uint32(report_precision, 2, "Report precision");

using namespace mooncake;

int g_deviceId = 0;
#define HOST_BUFFER_SIZE 0x1000

const static std::unordered_map<std::string, uint64_t> RATE_UNIT_MP = {
    {"GB", 1000ull * 1000ull * 1000ull},
    {"GiB", 1ull << 30},
    {"Gb", 1000ull * 1000ull * 1000ull / 8},
    {"MB", 1000ull * 1000ull},
    {"MiB", 1ull << 20},
    {"Mb", 1000ull * 1000ull / 8},
    {"KB", 1000ull},
    {"KiB", 1ull << 10},
    {"Kb", 1000ull / 8}};

static inline std::string calculateRate(uint64_t data_bytes, uint64_t duration) {
    if (!RATE_UNIT_MP.count(FLAGS_report_unit)) {
        LOG(WARNING) << "Invalid flag: report_unit only support "
                        "GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb, not support "
                     << FLAGS_report_unit
                     << " . Now use GB(default) as report_unit";
        FLAGS_report_unit = "GB";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(FLAGS_report_precision)
        << 1.0 * data_bytes * 1000000 / duration / RATE_UNIT_MP.at(FLAGS_report_unit)
        << " " << FLAGS_report_unit << "/s";
    return oss.str();
}

int device_malloc(void* &dev_addr, size_t size){
    // create acl
    aclError ret = aclrtMalloc(&dev_addr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to allocate host memory, ret:%d\n", ret);
        return -1;
    }

    // malloc mem
    void* host_addr = nullptr;
    ret = aclrtMallocHost(&host_addr, size);
    if (ret != ACL_ERROR_NONE || host_addr == NULL) {
        printf("Failed to allocate host memory, ret:%d\n", ret);
        aclrtFree(dev_addr);
        return -1;
    }

    // reg host mem
    for (size_t i = 0; i < size; i += sizeof(uint32_t)) {
        *(uint32_t*)((char *)host_addr + i) = 0x12345678;
    }

    // copy data from host mem to device mem
    ret = aclrtMemcpy(dev_addr, size, host_addr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to copy data from host to device, ret:%d\n", ret);
        aclrtFreeHost(host_addr);
        aclrtFree(dev_addr);
        return -1;
    }
    
    //release resource
    aclrtFreeHost(host_addr);

    return 0;
}

int initiator() {
    aclrtContext context = NULL;
    aclError ret = aclrtCreateContext(&context, g_deviceId);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to create context, ret:%d\n", ret);
        aclFinalize();
        return -1;
    }

    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(FLAGS_auto_discovery);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    std::string FLAGS_local_server_name_new = hostname_port.first + ":" + std::to_string(hostname_port.second) + ":npu_" + std::to_string(g_deviceId);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name_new.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);
    
    // 注册一块host内存，和vllm-connector场景保持一致Add commentMore actions
    void* host_addr = nullptr;
    ret = aclrtMallocHost(&host_addr, HOST_BUFFER_SIZE);
    if (ret != ACL_ERROR_NONE || host_addr == NULL) {
        printf("Failed to allocate host memory, ret:%d\n", ret);
        return -1;
    }

    ret = engine->registerLocalMemory(host_addr, HOST_BUFFER_SIZE, "cpu");
    void *dev_addr = NULL;
    std::vector<void *> g_addr;
    for (uint32_t i = 0; i < FLAGS_block_iteration + 1; i++ ) {
        uint64_t block_size = FLAGS_block_size * (1 << i);
        ret = device_malloc(dev_addr, FLAGS_batch_size * block_size);
        if (ret != 0) {
            return -1;
        }
        LOG(INFO) << "dev_addr_initor: " << dev_addr << " len:" << FLAGS_batch_size * block_size;
        int rc = engine->registerLocalMemory(dev_addr, FLAGS_batch_size * block_size,
                                            "npu:" + std::to_string(g_deviceId));
        LOG_ASSERT(!rc);
        g_addr.push_back(dev_addr);
    }

    auto segment_id = engine->openSegment(FLAGS_segment_id.c_str());

    TransferRequest::OpCode opcode;
    if (FLAGS_operation == "read")
        opcode = TransferRequest::READ;
    else if (FLAGS_operation == "write")
        opcode = TransferRequest::WRITE;
    else {
        LOG(ERROR) << "Unsupported operation: must be 'read' or 'write'";
        exit(EXIT_FAILURE);
    }

    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    if (!segment_desc) {
        LOG(ERROR) << "Unable to get target segment ID, please recheck";
        exit(EXIT_FAILURE);
    }

    Status s;
    uint64_t remote_base = 0;
    for (uint32_t i = 0; i < FLAGS_block_iteration + 1; i++) {
        uint64_t block_size = FLAGS_block_size * (1 << i);
        struct timeval start_tv, stop_tv;
        gettimeofday(&start_tv, nullptr);
        remote_base = (uint64_t)segment_desc->buffers[i + 1].addr;
        auto batch_id = engine->allocateBatchID(FLAGS_batch_size);
        std::vector<TransferRequest> requests;
        for (int j = 0; j < FLAGS_batch_size; ++j) {
            TransferRequest entry;
            entry.opcode = opcode;
            entry.length = block_size;
            entry.source = (uint8_t *)(g_addr[i]) + block_size * j;
            entry.target_id = segment_id;
            entry.target_offset = remote_base + block_size * j; 
            requests.emplace_back(entry);
        }
        s = engine->submitTransfer(batch_id, requests);
        LOG_ASSERT(s.ok());
        for (int task_id = 0; task_id < FLAGS_batch_size; ++task_id) {
            bool completed = false;
            TransferStatus status;
            while (!completed) {
                Status s = engine->getTransferStatus(batch_id, task_id, status);
                LOG_ASSERT(s.ok());
                if (status.s == TransferStatusEnum::COMPLETED)
                    completed = true;
                else if (status.s == TransferStatusEnum::FAILED) {
                    LOG(INFO) << "FAILED";
                    completed = true;
                    exit(EXIT_FAILURE);
                }
            }
        }
        gettimeofday(&stop_tv, nullptr);
        uint64_t duration = (stop_tv.tv_sec - start_tv.tv_sec) * 1000000.0 +
                        (stop_tv.tv_usec - start_tv.tv_usec);

        LOG(INFO) << "Test completed: duration " << duration << "us, block size "
                << block_size / 1024 << "KB, total size "
                << FLAGS_batch_size * block_size / 1024 << "KB , throughput "
                << calculateRate(
                        FLAGS_batch_size * block_size,
                        duration);
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }

    //release resource
    aclrtFreeHost(host_addr);
    for (uint32_t i = 0; i < FLAGS_block_iteration + 1; i++) {
        aclrtFree(g_addr[i]);
    }
    return 0;
}

volatile bool target_running = true;

int target() {
    aclrtContext context = NULL;
    aclError ret = aclrtCreateContext(&context, g_deviceId);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to create context\n");
        aclFinalize();
        return -1;
    }

    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(FLAGS_auto_discovery);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    std::string FLAGS_local_server_name_new = hostname_port.first + ":" + std::to_string(hostname_port.second) + ":npu_" + std::to_string(g_deviceId);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name_new.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);
    
    // 注册一块host内存，和vllm-connector场景保持一致Add commentMore actions
    void* host_addr = nullptr;
    ret = aclrtMallocHost(&host_addr, HOST_BUFFER_SIZE);
    if (ret != ACL_ERROR_NONE || host_addr == NULL) {
        printf("Failed to allocate host memory, ret:%d\n", ret);
        return -1;
    }

    ret = engine->registerLocalMemory(host_addr, HOST_BUFFER_SIZE, "cpu");

    void *dev_addr = NULL;
    std::vector<void *> g_addr;
    for (uint32_t i = 0; i < FLAGS_block_iteration + 1; i++) {
        uint64_t block_size = FLAGS_block_size * (1 << i);
        ret = device_malloc(dev_addr, FLAGS_batch_size * block_size);
        if (ret != 0) {
            return -1;
        }
        LOG(INFO) << "dev_addr_initor: " << dev_addr << ", len: " << FLAGS_batch_size * block_size;
        int rc = engine->registerLocalMemory(dev_addr, FLAGS_batch_size * block_size,
                                            "npu:" + std::to_string(g_deviceId));
        LOG_ASSERT(!rc);
        g_addr.push_back(dev_addr);
    }

    while (target_running) sleep(1);

    //release resource
    aclrtFreeHost(host_addr);
    for (uint32_t i = 0; i < FLAGS_block_iteration + 1; i++) {
        aclrtFree(g_addr[i]);
    }
    
    return 0;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    g_deviceId = FLAGS_device_id;
    //init ACL 
    const char *aclConfigPath = NULL;
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to initialize ACL\n");
        return -1;
    }

    ret = aclrtSetDevice(g_deviceId);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to set device ACL\n");
        return -1;
    }

    if (FLAGS_mode == "initiator") {
        return initiator();
    } else if (FLAGS_mode == "target") {
        return target();
    }

    LOG(ERROR) << "Unsupported mode: must be 'initiator' or 'target'";
    exit(EXIT_FAILURE);
}
