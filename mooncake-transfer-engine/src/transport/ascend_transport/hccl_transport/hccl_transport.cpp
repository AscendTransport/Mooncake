// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
#include <cassert>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cstdlib>
#include <string>
#include "transport/ascend_transport/hccl_transport/hccl_transport.h"

namespace mooncake{
HcclTransport::HcclTransport() : running_(-1) {
    //TODO
}

HcclTransport::~HcclTransport() {
    //TODO
    metadata_->removeSegmentDesc(local_server_name_);
}

void HcclTransport::initiatorLoop(int deviceLogicId, int selfIdx){
    aclrtStream stream;
    int ret = aclrtSetDevice(deviceLogicId);
    if (ret){
        LOG(ERROR) << "HcclTransport: aclrtSetDevice error, ret:" << ret;
        return;
    }

    ret = aclrtCreateStream(&stream);
    if (ret){
        LOG(ERROR) << "HcclTransport: aclrtCreateStream error, ret:" << ret;
        return;
    }
    while(1) {
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        if (allReqQueues_[selfIdx].empty()){
            initiator_cond_.wait(lock);
        }

        auto slice = std::move(allReqQueues_[selfIdx].front());
        allReqQueues_[selfIdx].pop();
        lock.unlock();
        auto segment_desc = metadata_->getSegmentDescByID(slice->target_id);
        if (!segment_desc) {
            LOG(ERROR) << "Unable to get target segment ID, please recheck";
            return;
        }

        remote_rank_info_.rankId = segment_desc->rank_info.rankId;
        inet_pton(AF_INET, segment_desc->rank_info.hostIp.c_str(), &remote_rank_info_.hostIp);
        remote_rank_info_.hostPort = segment_desc->rank_info.hostPort;
        remote_rank_info_.deviceLogicId = segment_desc->rank_info.deviceLogicId;
        remote_rank_info_.devicePhyId = segment_desc->rank_info.devicePhyId;
        inet_pton(AF_INET, segment_desc->rank_info.deviceIp.c_str(), &remote_rank_info_.deviceIp);
        remote_rank_info_.devicePort = segment_desc->rank_info.devicePort;
        remote_rank_info_.serverIdx = 0;

        ret = transportMemTask(&local_rank_info_, &remote_rank_info_, slice->opcode,
            slice->hccl.dest_addr, slice->length, slice->source_addr, stream);
        if (ret != HCCL_SUCCESS){
            LOG(ERROR) << "HcclTransport: transportMemTask error, ret:" << ret;
            slice->markFailed();
        } else {
            slice->markSuccess();
            slice->task->transferred_bytes = slice->length;
        }
    }
}

void HcclTransport::acceptLoop(int deviceLogicId){
    int ret = aclrtSetDevice(deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtSetDevice failed ret:" << ret;
        return;
    }
    while(1) {
        ret = transportMemAccept(&local_rank_info_);
        if (ret) {
            LOG(ERROR) << "HcclTransport: transportMemAccept failed ret:" << ret;
            return;
        } 
    }
}

int HcclTransport::initPdThread(){
    pid_t pid = getpid();

    int ret = 0;
    int deviceLogicId;
    ret = aclrtGetDevice(&deviceLogicId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: aclrtGetDevice failed ret:" << ret;
        return ret;
    } 

    for (int i = 0; i < THREAD_NUM; ++i) {
        allInitiatorThreads_[i] = std::thread(&HcclTransport::initiatorLoop, this, deviceLogicId, i);
        allAcceptThreads_[i] = std::thread(&HcclTransport::acceptLoop, this, deviceLogicId);
    }

    LOG(INFO) << "HcclTransport: initPdThread, pid: " << pid << ";" << "init " << THREAD_NUM << " initiator threads and accept threads, deviceLogicId: " << deviceLogicId;
    return 0;
}

// getrankid
int HcclTransport::getDevIdAndIpPortFromServerName(std::string& identifier, std::string& ip, int &port, int& npuId) {
    size_t firstColon = identifier.find(":");
    if (firstColon == std::string::npos) {
        LOG(ERROR) << "HcclTransport: getDevIdAndIpPortFromServerName failed, identifier is empty";
        return -1;
    }

    size_t secondColon = identifier.find(":", firstColon + 1);
    if (secondColon == std::string::npos) {
        LOG(ERROR) << "HcclTransport: getDevIdAndIpPortFromServerName failed, second colon missing";
        return -1;
    }

    ip = identifier.substr(0, firstColon);

    std::string portStr = identifier.substr(firstColon + 1, secondColon - firstColon - 1);
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

int HcclTransport::findDeviceInfo(const cJSON* root, int devicePhyId) {
    int deviceLogicId = 0;
    int ret = aclrtGetDevice(&deviceLogicId);
    if (ret != 0) {
        LOG(ERROR) << "HcclTransport: aclrtGetDevice failed." << ret;
        return ret;
    }
    LOG(INFO) << "deviceLogicId: "  << deviceLogicId << "devicePhyId: "  << devicePhyId;
    std::string devicePhyIdStr = std::to_string(devicePhyId);
    cJSON* serverList = cJSON_GetObjectItem(root, "server_list");
    if(!serverList || !cJSON_IsArray(serverList)) {
        LOG(ERROR) << "HcclTransport: Invalid JSON format: 'server_list' is missing or not an array.";
        return -1;
    }

    for (int i = 0; i < cJSON_GetArraySize(serverList); ++i){
        cJSON* server = cJSON_GetArrayItem(serverList, i);
        if (!server) continue;

        cJSON* serverId = cJSON_GetObjectItem(server, "server_id");
        if(!serverId || !cJSON_IsString(serverId)) {
            LOG(ERROR) << "HcclTransport: Invalid JSON format: 'server_id' is missing or not a string.";
            continue;
        }

        // make server_id to IP
        if(inet_pton(AF_INET, serverId->valuestring, &local_rank_info_.hostIp) != 1){
            LOG(ERROR) << "HcclTransport: Invalid IP format:" << serverId->valuestring;
            continue;
        }

        cJSON* deviceList = cJSON_GetObjectItem(server, "device");
        if(!deviceList || !cJSON_IsArray(deviceList)) {
            LOG(ERROR) << "HcclTransport: Invalid JSON format: 'device' is missing or not a array.";
            continue;
        }

        for(int j = 0; j < cJSON_GetArraySize(deviceList); ++j){
            cJSON* device = cJSON_GetArrayItem(deviceList, j);
            if (!device)continue;

            cJSON* deviceId = cJSON_GetObjectItem(device, "device_id");
            cJSON* deviceIp = cJSON_GetObjectItem(device, "device_ip");
            cJSON* rankId = cJSON_GetObjectItem(device, "rank_id");

            if(!deviceId || !cJSON_IsString(deviceId) ||
               !deviceIp || !cJSON_IsString(deviceIp) ||
               !rankId || !cJSON_IsString(rankId)) {
                LOG(ERROR) << "HcclTransport: Invalid JSON format: Missing 'deviceId', 'deviceIp', 'rankId'";
                continue;
            }

            if (deviceId->valuestring == devicePhyIdStr) {
                //make RankInfo struct
                const char* rankIdStr = rankId->valuestring;
                if (rankIdStr != NULL){
                    local_rank_info_.rankId = atoi(rankIdStr);
                }
                local_rank_info_.serverIdx = 0;
                local_rank_info_.devicePhyId = devicePhyId;
                local_rank_info_.hostPort = ASCEND_DEFAULT_HOST_PORT + devicePhyId;
                local_rank_info_.deviceLogicId = deviceLogicId;
                local_rank_info_.devicePort = ASCEND_DEFAULT_DEVICE_PORT;
                if (inet_pton(AF_INET, deviceIp->valuestring, &local_rank_info_.deviceIp) != 1) {
                    LOG(ERROR) << "HcclTransport: Invalid Device IP format: " << deviceIp->valuestring;
                    return -1;
                } 
                return 0;
            }
        }
    }

    return -1;
}

// parse rank table json
int HcclTransport::rankTableParse(int devicePhyId) {
    int ret = 0;
    const char* envRankTablePath = std::getenv("ENV_RANKTABLE_PATH");
    std::string filePath;
    if (!envRankTablePath) {
        LOG(INFO) << "Environment variables ENV_RANKTABLE_PATH are not set. use Default Path:/etc/hccl_16p.json";
        filePath = std::string("/etc/hccl_16p.json");
    } else {
        filePath = std::string(envRankTablePath);
    }

    std::ifstream file(filePath);
    ret = file.is_open();
    if (!ret) {
        LOG(ERROR) << "HcclTransport: Failed to open file: " << filePath << "ret: " << ret;
        return -1;
    }
    std::string jsonContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (jsonContent.empty()) {
        LOG(ERROR) << "HcclTransport: jsonContent is empty: " << filePath;
        return -1;
    }

    cJSON* root = cJSON_Parse(jsonContent.c_str());
    if(!root) {
        LOG(ERROR) << "HcclTransport: Failed to prase JSON:" << cJSON_GetErrorPtr();
        return -1;
    }

    ret = findDeviceInfo(root, devicePhyId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: Failed to findDeviceInfo: " << ret;
        return ret;
    }

    cJSON_Delete(root);
    return 0;
}

int HcclTransport::install(std::string &local_server_name,
                          std::shared_ptr<TransferMetadata> meta, std::shared_ptr<Topology> topo) {
    int ret = 0;
    int port;
    std::string ip;
    int devicePhyId;
    metadata_ = meta;
    ret = getDevIdAndIpPortFromServerName(local_server_name, ip, port, devicePhyId);
    if (ret < 0){
        LOG(ERROR) << "HcclTransport: getDevIdAndIpPortFromServerName failed, ret: " << ret;
        return ret; 
    }
    // 以ip:port作为desc_name
    local_server_name_ = ip + ":" + std::to_string(port);
    LOG(INFO) << "HcclTransport: local devicePhyId: " << devicePhyId  << ", local_server_name: " << local_server_name;

    // add to rankinfo_
    ret = rankTableParse(devicePhyId);
    if (ret) {
        LOG(ERROR) << "HcclTransport: rankTableParse failed, ret: " << ret;
        return ret;
    }

    ret = allocateLocalSegmentID();
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot allocate local segment, ret: "<< ret;
        return ret;
    }

    ret = metadata_->updateLocalSegmentDesc();
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot publish segments, "
                      "check the availability of metadata storage, ret: "<< ret;
        return ret;
    }

    ret = initTransportMem(&local_rank_info_);
    if (ret) {
        LOG(ERROR) << "HcclTransport: initTransportMem failed, ret: "<< ret;
        return ret;
    }

    ret = initPdThread();
    if (ret) {
        LOG(ERROR) << "HcclTransport: initPdThread failed, ret: "<< ret;
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
        slice->hccl.target_rank = 0;
        slice->status = Slice::PENDING;
        task.slice_list.push_back(slice);
        __sync_fetch_and_add(&task.slice_count, 1);
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        allReqQueues_[0].push(slice);
        lock.unlock();
        initiator_cond_.notify_one();
    }

    return Status::OK();
}

Status HcclTransport::submitTransferTask(
    const std::vector<TransferRequest *> &request_list,
    const std::vector<TransferTask *> &task_list) {
    for (size_t index = 0; index < request_list.size(); ++index) {
        auto &request = *request_list[index];
        auto &task = *task_list[index];
        task.total_bytes = request.length;
        Slice *slice = getSliceCache().allocate();
        slice->source_addr = (char *)request.source;
        slice->length = request.length;
        slice->opcode = request.opcode;
        slice->hccl.dest_addr = request.target_offset;
        slice->task = &task;
        slice->target_id = request.target_id;
        slice->hccl.target_rank = 0;
        slice->status = Slice::PENDING;
        task.slice_list.push_back(slice);
        __sync_fetch_and_add(&task.slice_count, 1);
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        allReqQueues_[0].push(slice);
        lock.unlock();
        initiator_cond_.notify_one();
    }

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
    buffer_desc.length = length;

    int ret;
    ret = regLocalRmaMem(addr, length);
    if(ret){
        LOG(ERROR) << "HcclTransport: reglocalRmaMem failed, ret:" << ret;
        return ret;
    }

    ret = metadata_->addLocalMemoryBuffer(buffer_desc, update_metadata);
    if(ret){
        LOG(ERROR) << "HcclTransport: addLocalMemoryBuffer failed,ret: " << ret;
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
    desc->rank_info.hostIp = inet_ntoa(local_rank_info_.hostIp);
    desc->rank_info.hostPort = local_rank_info_.hostPort;
    desc->rank_info.deviceLogicId = local_rank_info_.deviceLogicId;
    desc->rank_info.devicePhyId = local_rank_info_.devicePhyId;
    desc->rank_info.deviceIp = inet_ntoa(local_rank_info_.deviceIp);
    desc->rank_info.devicePort = local_rank_info_.devicePort;

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

}

