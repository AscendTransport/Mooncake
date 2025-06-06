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

static std::string getHostname();

DEFINE_string(local_server_name_initiator, getHostname(),
              "initiator Local server name for segment discovery");
DEFINE_string(local_server_name_target, getHostname(),
              "target Local server name for segment discovery");
DEFINE_string(metadata_server, "10.244.182.87:2379", "etcd server host address");
DEFINE_string(mode, "initiator",
              "Running mode: initiator or target. Initiator node read/write "
              "data blocks from target node");
DEFINE_string(operation, "read", "Operation type: read or write");

DEFINE_string(protocol, "hccl", "Transfer protocol: rdma|tcp|hccl");

DEFINE_string(segment_id, "192.168.3.76", "Segment ID to access data");
DEFINE_uint64(buffer_size, 524288, "total size of data buffer");
DEFINE_int32(batch_size, 10, "Batch size");
DEFINE_uint64(block_size, 4096, "Block size for each transfer request");
DEFINE_bool(auto_discovery, false, "Enable auto discovery");

DEFINE_uint64(device_id, 0, "The device ID of this machine");
DEFINE_string(segment_id_1, "NA", "A segment ID that a sender wants to another receiver");
DEFINE_uint64(recv_num, 1, "Num of coonections received by the receiver");
DEFINE_uint64(send_index, 0, "which one is sent to the same receiver");

using namespace mooncake;

int procSize = 0;
int procRank = 0;

static std::string getHostname() {
    char hostname[256];
    if (gethostname(hostname, 256)) {
        PLOG(ERROR) << "Failed to get hostname";
        return "";
    }
    return hostname;
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
    aclError ret = aclrtCreateContext(&context, procRank);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to create context, ret:%d\n", ret);
        aclFinalize();
        return -1;
    }

    sleep(1);

    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(FLAGS_auto_discovery);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name_initiator);
    int new_port = hostname_port.second + procRank;
    std::string FLAGS_local_server_name_new = hostname_port.first + ":" + std::to_string(new_port) + ":npu_" + std::to_string(procRank);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name_new.c_str(),
                 hostname_port.first.c_str(), new_port);

    void *dev_addr = NULL;
    device_malloc(dev_addr, FLAGS_block_size * FLAGS_batch_size);

    LOG(INFO) << "dev_addr_initor: " << dev_addr;

    int rc = engine->registerLocalMemory(dev_addr, FLAGS_buffer_size,
                                        "npu:" + std::to_string(procRank));
    LOG_ASSERT(!rc);

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
    uint64_t remote_base =
        (uint64_t)segment_desc->buffers[0].addr;   

    auto batch_id = engine->allocateBatchID(FLAGS_batch_size);
    Status s;
    std::vector<TransferRequest> requests;
    for (int i = 0; i < FLAGS_batch_size; ++i) {
        TransferRequest entry;
        entry.opcode = opcode;
        entry.length = FLAGS_block_size;
        entry.source = (uint8_t *)(dev_addr) + FLAGS_block_size * i;
        entry.target_id = segment_id;
        entry.target_offset = remote_base + FLAGS_block_size * i + FLAGS_buffer_size * FLAGS_send_index; 
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
    LOG(INFO) << "Send OK";
    s = engine->freeBatchID(batch_id);
    LOG_ASSERT(s.ok());

    if (FLAGS_segment_id_1 != "NA") {
        auto segment_id_1 = engine->openSegment(FLAGS_segment_id_1.c_str());

        TransferRequest::OpCode opcode;
        if (FLAGS_operation == "read")
            opcode = TransferRequest::READ;
        else if (FLAGS_operation == "write")
            opcode = TransferRequest::WRITE;
        else {
            LOG(ERROR) << "Unsupported operation: must be 'read' or 'write'";
            exit(EXIT_FAILURE);
        }    

        auto batch_id = engine->allocateBatchID(FLAGS_batch_size);
        Status s;
        std::vector<TransferRequest> requests;
        for (int i = 0; i < FLAGS_batch_size; ++i) {
            TransferRequest entry;
            entry.opcode = opcode;
            entry.length = FLAGS_block_size;
            entry.source = (uint8_t *)(dev_addr) + FLAGS_block_size * i;
            entry.target_id = segment_id_1;
            entry.target_offset = FLAGS_block_size * i;
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

        LOG(INFO) << "Send OK 2rd";
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }

    return 0;
}

volatile bool target_running = true;

int target() {
    aclrtContext context = NULL;
    aclError ret = aclrtCreateContext(&context, procRank);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to create context\n");
        aclFinalize();
        return -1;
    }

    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(FLAGS_auto_discovery);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name_target);
    int new_port = hostname_port.second + procRank;
    std::string FLAGS_local_server_name_new = hostname_port.first + ":" + std::to_string(new_port) + ":npu_" + std::to_string(procRank);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name_new.c_str(),
                 hostname_port.first.c_str(), new_port);

    void *dev_addr = NULL;
    device_malloc(dev_addr, FLAGS_block_size * FLAGS_batch_size);

    LOG(INFO) << "dev_addr_target: " << dev_addr;

    int rc = engine->registerLocalMemory(dev_addr, FLAGS_buffer_size * FLAGS_recv_num,
                                        "npu:" + std::to_string(procRank));
    LOG_ASSERT(!rc);

    while (target_running) sleep(1);
    
    return 0;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    procRank = FLAGS_device_id;
    //init ACL 
    const char *aclConfigPath = NULL;
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to initialize ACL\n");
        return -1;
    }

    ret = aclrtSetDevice(procRank);
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









