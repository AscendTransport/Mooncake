// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string>
#include "transport/ascend_transport/hccl_transport/hccl_transport.h"

#define READ 0
#define WRITE 1

namespace mooncake{
HcclTransport::HcclTransport() : running_(false) {
    //TODO
    baseTag_ = "ascend_transport";
}

HcclTransport::~HcclTransport() {
    //TODO
    metadata_->removeSegmentDesc(local_server_name_);
}

int HcclTransport::hcclInitTransportMem(int rank) {
    int deviceLogicId = rank;
    int devicePhyId = 0;
    int devicePort = 10000;
    HcclResult ret;

    //Init netdev
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId, deviceLogicId, false);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "HcclNetInit failed, ret: " << ret;
        return -1;
    }

    hccl::HcclIpAddress localVnicIp(devicePhyId);
    ret = hrtRaGetSingleSocketVnicIpInfo(devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId, localVnicIp);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "hrtRaGetSingleSocketVnicIpInfo failed, ret: " << ret;
        return -1;
    }

    ret = HcclNetOpenDev(&vnicNetDevCtx_, NicType::VNIC_TYPE, devicePhyId, deviceLogicId, localVnicIp);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "HcclNetOpenDev failed, ret: " << ret;
        return -1;
    }
    // control plane connection, creat serversocket, listening client
    vnicServerSocket_ = std::make_shared<hccl::HcclSocket>(vnicNetDevCtx_, devicePort);
    ret = vnicServerSocket_->Init();
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "vnicServerSocket_ Init failed, ret: " << ret;
        return -1;
    }
    ret = vnicServerSocket_->Listen();
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "vnicServerSocket_ Listen failed, ret: " << ret;
        return -1;
    }

    std::vector<unsigned int> remoteDevPhyId;
    int remote_devicePhyId = 0;
    remoteDevPhyId.push_back(remote_devicePhyId);
    HCCLCHECK(hccl::P2PMgmtPub::EnableP2P(remoteDevPhyId));
    HCCLCHECK(hccl::P2PMgmtPub::WaitP2PEnabled(remoteDevPhyId));

    hccl::HcclIpAddress remoteIp(devicePhyId);
    HCCLCHECK(hrtRaGetSingleSocketVnicIpInfo(remote_devicePhyId,
                                            DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                                            devicePhyId,
                                            remoteIp));

    std::vector<SocketWlistInfo> wlistInfoVec;
    SocketWlistInfo wlistInfo = {0};

    wlistInfo.connLimit = 1;
    memcpy(&wlistInfo.tag[0], baseTag_.c_str(), baseTag_.size() + 1);

    wlistInfo.remoteIp.addr = remoteIp.GetBinaryAddress().addr;
    wlistInfo.remoteIp.addr6 = remoteIp.GetBinaryAddress().addr6;
    wlistInfoVec.push_back(wlistInfo);
    ret = vnicServerSocket_->AddWhiteList(wlistInfoVec);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "vnicServerSocket_ AddWhiteList failed, ret: " << ret;
        return -1;
    }

    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId, &dispatcher_);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "HcclDispatcherInit failed, ret: " << ret;
        return -1;
    }

    notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
    if (notifyPool_ == nullptr) {
        LOG(ERROR) << "create notifyPool error";
        return -1;
    }
    ret = notifyPool_->Init(devicePhyId);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "Init notifyPool error, ret: " << ret;
        return -1;
    }
    notifyPool_->RegisterOp(baseTag_);
    return 0;
}

void HcclTransport::initiatorLoop(int deviceId, int selfIdx, int rank){

    aclrtSetDevice(deviceId);
    aclrtStream the_stream;
    aclrtCreateStream(&the_stream);
    int ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    while(1) {
        std::unique_lock<std::mutex> lock(initiator_mutex_);
        if (allReqQueues_[selfIdx].empty()){
            initiator_cond_.wait(lock);
        }
        auto slice = std::move(allReqQueues_[selfIdx].front());
        allReqQueues_[selfIdx].pop();
        lock.unlock();
        auto iter = target_rank_to_transport_map_.find(slice->hccl.target_rank);
        int remoteDevicePort = 0;
        if (iter == target_rank_to_transport_map_.end()){
            hccl::HcclIpAddress remoteIp(0);
            //from targetrank get remoteIp and devicePort
            std::shared_ptr<hccl::HcclSocket> hccl_socket = 
                std::make_shared<hccl::HcclSocket>(
                    baseTag_, vnicNetDevCtx_, remoteIp, remoteDevicePort,
                    hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
            ret = hccl_socket->Init();
            if (ret != HCCL_SUCCESS) {
                char deviceIp[64];
                inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
                LOG(ERROR) << "client hccl_socket init failed, target rank_id:" 
                           << slice->hccl.target_rank
                           << ", remoteIp:" << deviceIp 
                           << ", remote port:" << remoteDevicePort;
                return;
            }
            ret = hccl_socket->Connect();
            if (ret != HCCL_SUCCESS) {
                char deviceIp[64];
                inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
                LOG(ERROR) << "client hccl_socket Connect failed, target rank_id:" 
                           << slice->hccl.target_rank
                           << ", remoteIp:" << deviceIp 
                           << ", remote port:" << remoteDevicePort;
                return;
            }
            int devicePhyId = 0;
            hccl::HcclSocketStatus status;
            do {
                status = hccl_socket->GetStatus();
            } while (status != hccl::HcclSocketStatus::SOCKET_OK);
            ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId, &dispatcher_);
            if (ret != HCCL_SUCCESS) {
                char deviceIp[64];
                inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
                LOG(ERROR) << "client HcclDispatcherInit failed, target rank_id:" 
                           << slice->hccl.target_rank
                           << ", remoteIp:" << deviceIp 
                           << ", remote port:" << remoteDevicePort;
                return;
            }
            hccl::TransportMem::AttrInfo attrInfo;
            attrInfo.localRankId = 0;
            attrInfo.remoteRankId = 0;
            int deviceLogicId = 0;
            long long int sdid = 0;
            ret = hrtGetDeviceInfo(
                deviceLogicId,
                HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
                HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, sdid);
            if (ret != HCCL_SUCCESS) {
                char deviceIp[64];
                inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
                LOG(ERROR) << "client hrtGetDeviceInfo failed, target rank_id:" 
                           << slice->hccl.target_rank
                           << ", remoteIp:" << deviceIp 
                           << ", remote port:" << remoteDevicePort;
                return;
            }
            attrInfo.sdid = sdid;
            attrInfo.serverId = 0;
            transport_mem = hccl::TransportMem::Create(hccl::TransportMem::TpType::IPC, notifyPool_, vnicNetDevCtx_, dispatcher_, attrInfo);
            ret = transport_mem->SetSocket(hccl_socket);
            if (ret != HCCL_SUCCESS) {
                char deviceIp[64];
                inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
                LOG(ERROR) << "client SetSocket failed, target rank_id:" 
                           << slice->hccl.target_rank
                           << ", remoteIp:" << deviceIp 
                           << ", remote port:" << remoteDevicePort;
                return;
            }
            target_rank_to_transport_map_[slice->hccl.target_rank] = transport_mem;
        } else {
            transport_mem = target_rank_to_transport_map_[slice->hccl.target_rank];
        }
        auto start = std::chrono::high_resolution_clock::now();
        pid_t pid = getpid();

        hccl::TransportMem::RmaOpMem localMem;
        localMem.addr = slice->source_addr;
        localMem.size = slice->length;
        hccl::TransportMem::RmaOpMem remoteMem;
        remoteMem.addr = (void*)slice->hccl.dest_addr;
        remoteMem.size = slice->length;
        if (slice->opcode == WRITE){
            ret = transport_mem->Write(remoteMem, localMem, &the_stream);
        } else {
            ret = transport_mem->Read(localMem, remoteMem, &the_stream);
        }

        if (ret != HCCL_SUCCESS){
            slice->markFailed();
        } else {
            slice->markSuccess();
            slice->task->transferred_bytes = slice->length;
        }
        auto mid = std::chrono::high_resolution_clock::now();
        transport_mem->AddOpFence(the_stream);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration_sync = std::chrono::duration_cast<std::chrono::milliseconds>(mid-start);
        auto duration_call = std::chrono::duration_cast<std::chrono::milliseconds>(stop-mid);
        LOG(INFO) << "pid: " << pid << "; " << "thread submit one block size: "<< slice->length;
        LOG(INFO) << "pid: " << pid << "; " << "thread sync stream spent: "<< duration_sync.count() << "ms";
        LOG(INFO) << "pid: " << pid << "; " << "thread call write/read spent: "<< duration_call.count() << "ms";

    }
}

void HcclTransport::acceptLoop(int deviceId, int selfIdx, int rank){
    while(1) {
        std::shared_ptr<hccl::HcclSocket> hccl_socket;
        int ret = vnicServerSocket_->Accept(baseTag_, hccl_socket);
        if (ret < 0) {
            LOG(ERROR) << "vnicServerSocket_ Accept failed ret:" << ret;
            return;
        } else {
            serverSocketVec_.push_back(hccl_socket);
        }
    }
}

void HcclTransport::initPdThread(int rank){
    pid_t pid = getpid();

    int deviceId;
    aclrtGetDevice(&deviceId);
    for (int i = 0; i < THREAD_NUM; ++i) {
        allInitiatorThreads_[i] = std::thread(&HcclTransport::initiatorLoop, this, deviceId, i, rank);
        allAcceptThreads_[i] = std::thread(&HcclTransport::acceptLoop, this, deviceId, i, rank);
    }
    LOG(INFO) << "pid: " << pid << ";" << "init " << THREAD_NUM << " initiator threads and accept threads";
}

// getrankid
int HcclTransport::getRankFromServerName(const std::string& local_server_name) {
    size_t underscore_pos = local_server_name.find_last_of('_');
    if (underscore_pos == std::string::npos) {
        LOG(ERROR) << "Invalid format: No underscore found.";
        return -1;
    }

    std::string number_part = local_server_name.substr(underscore_pos + 1);

    int rank_id = std::stoi(number_part);

    return rank_id;
}

int HcclTransport::install(std::string &local_server_name,
                          std::shared_ptr<TransferMetadata> meta, std::shared_ptr<Topology> topo) {
    metadata_ = meta;
    local_server_name_ = local_server_name;
    int rank = getRankFromServerName(local_server_name);

    int ret = allocateLocalSegmentID(rank);
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot allocate local segment";
        return -1;
    }

    ret = metadata_->updateLocalSegmentDesc();
    if (ret) {
        LOG(ERROR) << "HcclTransport: cannot publish segments, "
                      "check the availability of metadata storage";
        return -1;
    }

    hcclInitTransportMem(rank);
    initPdThread(rank);
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
        auto desc = metadata_->getSegmentDescByID(request.target_id);
        uint32_t target_rank = (uint32_t)desc->devices[0].lid;
        ++task_id;
        task.total_bytes = request.length;
        Slice *slice = getSliceCache().allocate();
        slice->source_addr = request.source;
        slice->length = request.length;
        slice->opcode = request.opcode;
        slice->hccl.dest_addr = request.target_offset;
        slice->task = &task;
        slice->target_id = request.target_id;
        slice->hccl.target_rank = target_rank;
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
    for (int index = 0; index < request_list.size(); ++index) {
        auto &request = *request_list[index];
        auto &task = *task_list[index];
        auto desc = metadata_->getSegmentDescByID(request.target_id);
        uint32_t target_rank = (uint32_t)desc->devices[0].lid;
        task.total_bytes = request.length;
        Slice *slice = getSliceCache().allocate();
        slice->source_addr = (char *)request.source;
        slice->length = request.length;
        slice->opcode = request.opcode;
        slice->hccl.dest_addr = request.target_offset;
        slice->task = &task;
        slice->target_id = request.target_id;
        slice->hccl.target_rank = target_rank;
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
    const int task_count = batch_desc.task_list.size();
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
    return metadata_->addLocalMemoryBuffer(buffer_desc, update_metadata);
}

int HcclTransport::unregisterLocalMemory(void *addr, bool update_metadata) {
    return metadata_->removeLocalMemoryBuffer(addr, update_metadata);
}

int HcclTransport::allocateLocalSegmentID(int rank) {
    auto desc = std::make_shared<SegmentDesc>();
    if (!desc) return ERR_MEMORY;
    desc->name = local_server_name_;
    desc->protocol = "ascend";
    TransferMetadata::DeviceDesc device_desc;
    device_desc.name = "npu:" + std::to_string(rank);
    device_desc.lid = (uint16_t)rank;
    desc->devices.push_back(device_desc);

    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));
    return 0;
}

int HcclTransport::registerLocalMemoryBatch(
    const std::vector<Transport::BufferEntry> &buffer_list,
    const std::string &location) {
    for (auto &buffer : buffer_list)
        registerLocalMemory(buffer.addr, buffer.length, location, true, false);
    return metadata_->updateLocalSegmentDesc();
}

int HcclTransport::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    for (auto &addr : addr_list) unregisterLocalMemory(addr, false);
    return metadata_->updateLocalSegmentDesc();
}

}

