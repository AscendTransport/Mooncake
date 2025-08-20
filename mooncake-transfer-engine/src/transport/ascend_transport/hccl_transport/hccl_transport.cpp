// Copyright 2025 Huawei Technologies Co., Ltd
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

#include <cassert>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cstdlib>
#include <string>
#include "transport/ascend_transport/hccl_transport/hccl_transport.h"

namespace mooncake {
HcclTransport::HcclTransport() : running_(-1) {
    // TODO
}

HcclTransport::~HcclTransport() {
    if (running_) {
        running_ = false;
        initiator_cond_.notify_all();

        for (size_t i = 0; i < THREAD_NUM; ++i) {
            allInitiatorThreads_[i].join();
            allAcceptThreads_[i].join();
        }
    }
    metadata_->removeSegmentDesc(local_server_name_);
}

void HcclTransport::initiatorLoop(int deviceLogicId, int selfIdx) {
    aclrtStream stream;
    int ret = aclrtSetDevice(deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtSetDevice error, ret: " << ret;
    }

    ret = aclrtCreateStream(&stream);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtCreateStream error, ret: " << ret;
    }

    while (running_) {
        auto waitlock = std::chrono::high_resolution_clock::now();
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        initiator_cond_.wait(lock, [this, selfIdx]() {
            return !allReqQueues_[selfIdx].empty() || !running_;
        });

        if (!running_) {
            break;
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto slice_list = std::move(allReqQueues_[selfIdx].front());
        allReqQueues_[selfIdx].pop();
        lock.unlock();
        if (slice_list.empty()) {
            LOG(ERROR) << "HcclTransport: empty transfer request batch";
        }
        auto segment_desc =
            metadata_->getSegmentDescByID(slice_list[0]->target_id);
        if (!segment_desc) {
            LOG(ERROR) << "Unable to get target segment ID, please recheck, "
                          "segment ID: "
                       << slice_list[0]->target_id;
            for (auto slice : slice_list) {
                slice->markFailed();
            }
            continue;
        }

        remote_rank_info_.rankId = segment_desc->rank_info.rankId;
        remote_rank_info_.hostPort = segment_desc->rank_info.hostPort;
        remote_rank_info_.deviceLogicId = segment_desc->rank_info.deviceLogicId;
        remote_rank_info_.devicePhyId = segment_desc->rank_info.devicePhyId;
        remote_rank_info_.devicePort = segment_desc->rank_info.devicePort;
        remote_rank_info_.serverId = segment_desc->rank_info.serverId;
        remote_rank_info_.devPid = segment_desc->rank_info.devPid;
        remote_rank_info_.sdid = segment_desc->rank_info.sdid;

        strncpy(remote_rank_info_.hostIp,
                segment_desc->rank_info.hostIp.c_str(), 127);
        remote_rank_info_.hostIp[127] = '\0';

        strncpy(remote_rank_info_.deviceIp,
                segment_desc->rank_info.deviceIp.c_str(), 127);
        remote_rank_info_.deviceIp[127] = '\0';

        strncpy(remote_rank_info_.vnicIp,
                segment_desc->rank_info.vnicIp.c_str(), 127);
        remote_rank_info_.vnicIp[127] = '\0';

        for (auto slice : slice_list) {
            ret = transportMemTask(&local_rank_info_, &remote_rank_info_,
                                   slice->opcode, slice->hccl.dest_addr,
                                   slice->length, slice->source_addr, stream);
            if (ret) {
                LOG(ERROR) << "HcclTransport: transportMemTask error, local "
                              "devicePhyId: "
                           << local_rank_info_.devicePhyId
                           << ", remote devicePhyId: "
                           << remote_rank_info_.devicePhyId
                           << ", source_addr: " << slice->source_addr
                           << ", dest_addr: " << slice->hccl.dest_addr
                           << ", ret: " << ret;
                slice->markFailed();
                slice->status = Slice::SliceStatus::FAILED;
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        ret = transportMemAddOpFence(&remote_rank_info_, stream);
        if (ret) {
            LOG(ERROR) << "transportMemAddOpFence failed, local devicePhyId: "
                       << local_rank_info_.devicePhyId
                       << ", remote devicePhyId: "
                       << remote_rank_info_.devicePhyId << ", ret: " << ret;
            for (auto slice : slice_list) {
                slice->markFailed();
            }
        }
        auto addOpfence = std::chrono::high_resolution_clock::now();

        ret = aclrtSynchronizeStream(stream);
        if (ret) {
            LOG(ERROR) << "aclrtSynchronizeStream failed, local devicePhyId: "
                       << local_rank_info_.devicePhyId
                       << ", remote devicePhyId: "
                       << remote_rank_info_.devicePhyId << ", ret: " << ret;
            for (auto slice : slice_list) {
                slice->markFailed();
            }
        }
        for (auto slice : slice_list) {
            if (slice->status != Slice::SliceStatus::FAILED) {
                slice->markSuccess();
                slice->task->transferred_bytes = slice->length;
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        if (ascendPrintEnabled()) {
            pid_t pid = getpid();
            auto duration_wait =
                std::chrono::duration_cast<std::chrono::milliseconds>(start -
                                                                      waitlock);
            auto duration_call =
                std::chrono::duration_cast<std::chrono::microseconds>(mid -
                                                                      start);
            auto duration_addOpfence =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    addOpfence - mid);
            auto duration_sync =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop - addOpfence);
            LOG(INFO) << "pid: " << pid << ", target hostIp: "
                      << segment_desc->rank_info.hostIp.c_str()
                      << ", local devicePhyId: " << local_rank_info_.devicePhyId
                      << ", target devicePhyId: "
                      << remote_rank_info_.devicePhyId
                      << ", batch waitlock spent: " << duration_wait.count()
                      << "ms"
                      << ", batch call spent: " << duration_call.count() << "us"
                      << ", batch addOpfence spent: "
                      << duration_addOpfence.count() << "us"
                      << ", batch sync spent: " << duration_sync.count()
                      << "us";
        } else {
            (void)waitlock;
            (void)start;
            (void)mid;
            (void)addOpfence;
            (void)stop;
        }
    }
}

void HcclTransport::acceptLoop(int deviceLogicId) {
    int ret = aclrtSetDevice(deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtSetDevice failed ret: " << ret;
    }
    while (running_) {
        ret = transportMemAccept(&local_rank_info_);
        if (ret) {
            LOG(ERROR) << "HcclTransport: transportMemAccept failed ret: "
                       << ret;
        }
    }
}

int HcclTransport::initPdThread() {
    pid_t pid = getpid();
    int ret = 0;
    int deviceLogicId;
    ret = aclrtGetDevice(&deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtGetDevice failed, ret: " << ret;
        return ret;
    }

    for (int i = 0; i < THREAD_NUM; ++i) {
        allInitiatorThreads_[i] =
            std::thread(&HcclTransport::initiatorLoop, this, deviceLogicId, i);
        allAcceptThreads_[i] =
            std::thread(&HcclTransport::acceptLoop, this, deviceLogicId);
    }

    LOG(INFO) << "HcclTransport: initPdThread, pid: " << pid << ";" << "init "
              << THREAD_NUM
              << " initiator threads and accept threads, deviceLogicId: "
              << deviceLogicId;
    return 0;
}

// Get HostIp\Port\DevicePhyId
int HcclTransport::getDevIdAndIpPortFromServerName(std::string &identifier,
                                                   std::string &hostIp,
                                                   int &port, int &npuId) {
    size_t firstColon = identifier.find(":");
    if (firstColon == std::string::npos) {
        LOG(ERROR) << "HcclTransport: getDevIdAndIpPortFromServerName failed, "
                      "identifier is empty";
        return -1;
    }

    size_t secondColon = identifier.find(":", firstColon + 1);
    if (secondColon == std::string::npos) {
        LOG(ERROR) << "HcclTransport: getDevIdAndIpPortFromServerName failed, "
                      "second colon missing";
        return -1;
    }

    hostIp = identifier.substr(0, firstColon);

    std::string portStr =
        identifier.substr(firstColon + 1, secondColon - firstColon - 1);
    try {
        port = std::stoi(portStr);
    } catch (const std::exception &e) {
        LOG(ERROR) << "Invalid Port Number";
        return -1;
    }

    std::string npuStr = identifier.substr(secondColon + 1);
    if (npuStr.find("npu_") != 0) {
        LOG(ERROR) << "Invalid npu number format - should start with 'npu_'";
        return -1;
    }

    try {
        npuId = std::stoi(npuStr.substr(4));
    } catch (const std::exception &e) {
        LOG(ERROR) << "Invalid device_id ID";
        return -1;
    }

    return 0;
}

int HcclTransport::devInfoParse(std::string hostIp) {
    int deviceLogicId = 0;
    int ret = aclrtGetDevice(&deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtGetDevice failed, ret: " << ret;
        return ret;
    }
    local_rank_info_.deviceLogicId = (uint32_t)deviceLogicId;

    local_rank_info_.rankId = local_rank_info_.deviceLogicId;

    strncpy(local_rank_info_.hostIp, hostIp.c_str(), 127);
    local_rank_info_.hostIp[127] = '\0';

    local_rank_info_.devicePort = ASCEND_DEFAULT_DEVICE_PORT;

    local_rank_info_.hostPort =
        ASCEND_DEFAULT_HOST_PORT + local_rank_info_.devicePhyId;

    return 0;
}

int HcclTransport::install(std::string &local_server_name,
                           std::shared_ptr<TransferMetadata> meta,
                           std::shared_ptr<Topology> topo) {
    int port;
    std::string hostIp;
    int devicePhyId;
    metadata_ = meta;
    int ret = getDevIdAndIpPortFromServerName(local_server_name, hostIp, port,
                                              devicePhyId);
    if (ret) {
        LOG(ERROR)
            << "HcclTransport: getDevIdAndIpPortFromServerName failed, ret: "
            << ret;
        return ret;
    }

    local_server_name_ = hostIp + ":" + std::to_string(port);
    LOG(INFO)
        << "HcclTransport: begin to install transport, local devicePhyId: "
        << devicePhyId << ", local_server_name: " << local_server_name;

    // add to local_rank_info_
    ret = devInfoParse(hostIp);
    if (ret) {
        LOG(ERROR) << "HcclTransport: devInfoParse failed, ret: " << ret;
        return ret;
    }

    ret = initTransportMem(&local_rank_info_);
    if (ret) {
        LOG(ERROR) << "HcclTransport: initTransportMem failed, ret: " << ret;
        return ret;
    }

    ret = allocateLocalSegmentID();
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot allocate local segment, ret: "
                   << ret;
        return ret;
    }

    ret = metadata_->updateLocalSegmentDesc();
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot publish segments, "
                      "check the availability of metadata storage, ret: "
                   << ret;
        return ret;
    }

    ret = initPdThread();
    if (ret) {
        LOG(ERROR) << "HcclTransport: initPdThread failed, ret: " << ret;
        return ret;
    }

    return 0;
}

Status HcclTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    auto &batch_desc = *((BatchDesc *)(batch_id));
    if (batch_desc.task_list.size() + entries.size() > batch_desc.batch_size) {
        LOG(ERROR) << "HcclTransport: Exceed the limitation of current batch's "
                      "capacity";
        return Status::InvalidArgument(
            "HcclTransport: Exceed the limitation of capacity, batch id: " +
            std::to_string(batch_id));
    }

    int task_id = batch_desc.task_list.size();
    batch_desc.task_list.resize(task_id + entries.size());
    std::vector<Slice *> slice_list;
    slice_list.reserve(entries.size());
    for (auto &request : entries) {
        TransferTask &task = batch_desc.task_list[task_id];
        ++task_id;
        task.total_bytes = request.length;
        Slice *slice = getSliceCache().allocate();
        slice->source_addr = request.source;
        slice->length = request.length;
        slice->opcode = request.opcode;
        slice->hccl.dest_addr = request.target_offset;
        slice->task = &task;
        slice->target_id = request.target_id;
        slice->status = Slice::PENDING;
        task.slice_list.push_back(slice);
        __sync_fetch_and_add(&task.slice_count, 1);
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        slice_list.push_back(slice);
        lock.unlock();
        initiator_cond_.notify_one();
    }
    std::unique_lock<std::mutex> lock(initiator_mutex_);
    allReqQueues_[0].push(slice_list);
    lock.unlock();
    initiator_cond_.notify_one();

    return Status::OK();
}

Status HcclTransport::submitTransferTask(
    const std::vector<TransferTask *> &task_list) {
    std::vector<Slice *> slice_list;
    slice_list.reserve(task_list.size());
    for (size_t index = 0; index < task_list.size(); ++index) {
        assert(task_list[index]);
        auto &task = *task_list[index];
        assert(task.request);
        auto &request = *task.request;
        task.total_bytes = request.length;
        Slice *slice = getSliceCache().allocate();
        slice->source_addr = (char *)request.source;
        slice->length = request.length;
        slice->opcode = request.opcode;
        slice->hccl.dest_addr = request.target_offset;
        slice->task = &task;
        slice->target_id = request.target_id;
        slice->status = Slice::PENDING;
        slice->ts = 0;
        task.slice_list.push_back(slice);
        __sync_fetch_and_add(&task.slice_count, 1);
        slice_list.push_back(slice);
    }
    std::unique_lock<std::mutex> lock(initiator_mutex_);
    allReqQueues_[0].push(slice_list);
    lock.unlock();
    initiator_cond_.notify_one();

    return Status::OK();
}

Status HcclTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                        TransferStatus &status) {
    auto &batch_desc = *((BatchDesc *)(batch_id));
    const size_t task_count = batch_desc.task_list.size();
    if (task_id >= task_count) {
        return Status::InvalidArgument(
            "HcclTransport::getTransportStatus invalid argument, batch id: " +
            std::to_string(batch_id));
    }
    auto &task = batch_desc.task_list[task_id];
    status.transferred_bytes = task.transferred_bytes;
    uint64_t success_slice_count = task.success_slice_count;
    uint64_t failed_slice_count = task.failed_slice_count;
    if (success_slice_count + failed_slice_count == task.slice_count) {
        if (failed_slice_count) {
            status.s = TransferStatusEnum::FAILED;
        } else {
            status.s = TransferStatusEnum::COMPLETED;
        }
        task.is_finished = true;
    } else {
        status.s = TransferStatusEnum::WAITING;
    }
    return Status::OK();
}

int HcclTransport::registerLocalMemory(void *addr, size_t length,
                                       const std::string &location,
                                       bool remote_accessible,
                                       bool update_metadata) {
    (void)remote_accessible;
    BufferDesc buffer_desc;
    buffer_desc.name = location;
    buffer_desc.addr = (uint64_t)addr;
    buffer_desc.length = (uint64_t)length;

    int ret;
    ret = regLocalRmaMem(addr, (uint64_t)length);
    if (ret) {
        LOG(ERROR) << "HcclTransport: reglocalRmaMem failed, ret: " << ret;
        return ret;
    }

    ret = metadata_->addLocalMemoryBuffer(buffer_desc, update_metadata);
    if (ret) {
        LOG(ERROR) << "HcclTransport: addLocalMemoryBuffer failed, ret: "
                   << ret;
        return ret;
    }

    return 0;
}

int HcclTransport::unregisterLocalMemory(void *addr, bool update_metadata) {
    return metadata_->removeLocalMemoryBuffer(addr, update_metadata);
}

int HcclTransport::allocateLocalSegmentID() {
    auto desc = std::make_shared<SegmentDesc>();
    if (!desc) return ERR_MEMORY;
    desc->name = local_server_name_;
    desc->protocol = "ascend";
    desc->rank_info.rankId = local_rank_info_.rankId;
    desc->rank_info.hostIp = local_rank_info_.hostIp;
    desc->rank_info.hostPort = local_rank_info_.hostPort;
    desc->rank_info.deviceLogicId = local_rank_info_.deviceLogicId;
    desc->rank_info.devicePhyId = local_rank_info_.devicePhyId;
    desc->rank_info.deviceIp = local_rank_info_.deviceIp;
    desc->rank_info.devicePort = local_rank_info_.devicePort;
    desc->rank_info.devPid = local_rank_info_.devPid;
    desc->rank_info.sdid = local_rank_info_.sdid;
    desc->rank_info.serverId = local_rank_info_.serverId;
    desc->rank_info.vnicIp = local_rank_info_.vnicIp;

    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));
    return 0;
}

int HcclTransport::registerLocalMemoryBatch(
    const std::vector<Transport::BufferEntry> &buffer_list,
    const std::string &location) {
    for (auto &buffer : buffer_list)
        registerLocalMemory(buffer.addr, buffer.length, location, true, -1);
    return metadata_->updateLocalSegmentDesc();
}

int HcclTransport::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    for (auto &addr : addr_list) unregisterLocalMemory(addr, -1);
    return metadata_->updateLocalSegmentDesc();
}

}  // namespace mooncake
