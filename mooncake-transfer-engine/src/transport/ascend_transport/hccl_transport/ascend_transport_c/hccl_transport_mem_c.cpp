// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string>
#include "transport/ascend_transport/hccl_transport/hccl_transport_mem_c.h"

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#define READ 0
#define WRITE 1
#define UINT32MAX 1000

std::condition_variable initiator_cond_;
std::string baseTag_ = "transport_mem";
std::unique_ptr<hccl::NotifyPool> notifyPool_;
HcclNetDevCtx vnicNetDevCtx_{nullptr};
HcclDispatcher dispatcher_{nullptr};

std::vector<std::shared_ptr<hccl::HcclSocket>> clientSocketVec_;
std::shared_ptr<hccl::HcclSocket> vnicServerSocket_{nullptr};

std::unordered_map<std::string, int>  target_key_to_control_socket_map_;
std::unordered_map<std::string, std::shared_ptr<hccl::TransportMem>> 
    target_key_to_transport_mem_map_;
std::vector<hccl::TransportMem::RmaMem *> localRmaMem_;

std::vector<void *> g_mem_c;
std::vector<uint64_t> g_len_c;

int g_server_socket_ = 0;
struct sockaddr_in g_server_addr_;

static int initServerNetSocket(RankInfo *local_rank_info){
    //Init netdev
    int ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, local_rank_info->devicePhyId, 
                      local_rank_info->deviceLogicId, false);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "HcclNetInit failed, ret: " << ret;
        return -1;
    }

    hccl::HcclIpAddress localVnicIp(local_rank_info->devicePhyId);
    ret = hrtRaGetSingleSocketVnicIpInfo(
        local_rank_info->devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, 
        local_rank_info->devicePhyId, localVnicIp);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "hrtRaGetSingleSocketVnicIpInfo failed, ret: " << ret;
        return -1;
    }

    ret = HcclNetOpenDev(&vnicNetDevCtx_, NicType::VNIC_TYPE, 
                         local_rank_info->devicePhyId, 
                         local_rank_info->deviceLogicId, localVnicIp);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "HcclNetOpenDev failed, ret: " << ret;
        return -1;
    }

    // control plane connection, creat serversocket, listening client
    vnicServerSocket_ = std::make_shared<hccl::HcclSocket>(
        vnicNetDevCtx_, local_rank_info->devicePort);
    if (vnicServerSocket_ == NULL) {
        LOG(ERROR) << "vnicServerSocket_ make failed";
        return -1;
    }

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
    return 0;
}

static int initControlSocket(RankInfo *local_rank_info) {
    // control plane init
    g_server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_socket_ < 0) {
        LOG(ERROR) << "Socket create failed";
        return -1;
    }

    int optval = 1;
    if (setsockopt(g_server_socket_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
        LOG(ERROR) << "set sock opt failed";
        close(g_server_socket_);
        return -1;
    }

    memset(&g_server_addr_, 0, sizeof(g_server_addr_));
    g_server_addr_.sin_family = AF_INET;
    // listen on all network interfaces
    g_server_addr_.sin_addr.s_addr = INADDR_ANY;
    // unique port
    g_server_addr_.sin_port = htons(local_rank_info->hostPort);

    if (bind(g_server_socket_, (struct sockaddr*)&g_server_addr_, sizeof(g_server_addr_)) < 0) {
        LOG(ERROR) << "Bind Failed";
        close(g_server_socket_);
        return -1;
    }

    if (listen(g_server_socket_, UINT32MAX) < 0) {
        LOG(ERROR) << "Listen Failed";
        close(g_server_socket_);
        return -1;
    }
    LOG(INFO) << "initControlSocket successful, Server listening on port" << g_server_addr_.sin_port << "..." << "g_server_socket_" << g_server_socket_;
    return 0;
}

int initTransportMem(RankInfo *local_rank_info) {
    if (local_rank_info == NULL){
        LOG(ERROR) << "initTransportMem local_rank_info is NULL";
        return -1;
    }
    LOG(INFO) << "initTransportMem local_rank_info rankId: "
              << local_rank_info->rankId
              << ", serverIdx: " << local_rank_info->serverIdx
              << ", deviceLogicId: " << local_rank_info->deviceLogicId
              << ", devicePhyId: " << local_rank_info->devicePhyId
              << ", deviceIp: " << inet_ntoa(local_rank_info->deviceIp)
              << ", devicePort: " << local_rank_info->devicePort
              << ", hostIp: " << inet_ntoa(local_rank_info->hostIp)
              << ", hostPort: " << local_rank_info->hostPort;

        // 初始化数据通道虚拟网卡和socket，交换RmaMem，以及创建QP链接
        if (initServerNetSocket(local_rank_info) != 0) {
            return -1;
        }
        
        if (initControlSocket(local_rank_info) != 0) {
            return -1;
        }
        return 0;
}

static int connectToTarget(std::string target_ip, int target_port) {
    int client_socket;
    struct sockaddr_in server_addr;

    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        LOG(ERROR) << "Socket creation failed";
        return -1;
    }

    int optval = 1;
    setsockopt(client_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(target_port);
    server_addr.sin_addr.s_addr = inet_addr(target_ip.c_str());

    if (server_addr.sin_addr.s_addr == INADDR_NONE) {
        LOG(ERROR) << "Invalid server IP address";
        close(client_socket);
        return -1;
    }

    while (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {}

    LOG(INFO) << "Connect to server" << target_ip << ":" << target_port;

    return client_socket;
}

int transportMemTask(RankInfo *local_rank_info, RankInfo *remote_rank_info, 
                    int op_code, uint64_t offset,
                    uint64_t req_len, void *local_mem, aclrtStream stream)
{
    LOG(INFO) << "transportMemTask local_rank_info rankId: "
              << local_rank_info->rankId
              << ", serverIdx: " << local_rank_info->serverIdx
              << ", deviceLogicId: " << local_rank_info->deviceLogicId
              << ", devicePhyId: " << local_rank_info->devicePhyId
              << ", deviceIp: " << inet_ntoa(local_rank_info->deviceIp)
              << ", devicePort: " << local_rank_info->devicePort
              << ", hostIp: " << inet_ntoa(local_rank_info->hostIp)
              << ", hostPort: " << local_rank_info->hostPort;
    LOG(INFO) << "transportMemTask remote_rank_info rankId: "
              << remote_rank_info->rankId
              << ", serverIdx: " << remote_rank_info->serverIdx
              << ", deviceLogicId: " << remote_rank_info->deviceLogicId
              << ", devicePhyId: " << remote_rank_info->devicePhyId
              << ", deviceIp: " << inet_ntoa(remote_rank_info->deviceIp)
              << ", devicePort: " << remote_rank_info->devicePort
              << ", hostIp: " << inet_ntoa(remote_rank_info->hostIp)
              << ", hostPort: " << remote_rank_info->hostPort;

    // 1、封装控制信息
    RankControlInfo control_info;
    control_info.deviceLogicId = local_rank_info->deviceLogicId;
    control_info.devicePhyId = local_rank_info->devicePhyId;
    control_info.hostIp = local_rank_info->hostIp;
    // 2、查找对端，检查是否具备对应的socket，并发送本端的信息给对端
    std::string key_str = inet_ntoa(remote_rank_info->hostIp) + std::to_string(remote_rank_info->devicePhyId);
    auto iter = target_key_to_control_socket_map_.find(key_str);
    if (iter == target_key_to_control_socket_map_.end()) {
        int client_socket = connectToTarget(inet_ntoa(remote_rank_info->hostIp), 
                                            remote_rank_info->hostPort);
        if (client_socket < 0) {
            LOG(ERROR) << "client connect failed";
            return -1;
        }
        if (send(client_socket, &control_info, sizeof(RankControlInfo), 0) < 0) {
            LOG(ERROR) << "send control_info failed";
            close(client_socket);
            return -1;
        }
        target_key_to_control_socket_map_[key_str] = client_socket;
    }
    // 3、查找对端，检查时候具备对应的transport_mem,并发送本端的信息给对端。
    int ret = 0;
    int remoteDevicePort = remote_rank_info->devicePort;
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    auto iter_mem = target_key_to_transport_mem_map_.find(key_str);
    if (iter_mem == target_key_to_transport_mem_map_.end()) {
        std::vector<unsigned int> remoteDevPhyId;
        remoteDevPhyId.push_back(remote_rank_info->devicePhyId);
        HCCLCHECK(hccl::P2PMgmtPub::EnableP2P(remoteDevPhyId));
        HCCLCHECK(hccl::P2PMgmtPub::WaitP2PEnabled(remoteDevPhyId));

        hccl::HcclIpAddress remoteIp(remote_rank_info->devicePhyId);
        HCCLCHECK(hrtRaGetSingleSocketVnicIpInfo(local_rank_info->devicePhyId,
                                            DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                                            remote_rank_info->devicePhyId, remoteIp));
        std::shared_ptr<hccl::HcclSocket> hccl_socket = 
            std::make_shared<hccl::HcclSocket>(
                baseTag_, vnicNetDevCtx_, remoteIp, remoteDevicePort,
                hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
        ret = hccl_socket->Init();
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client hccl_socket init failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", remote port:" << remoteDevicePort
                        << ", ret: " << ret;
            return -1;
        }

        ret = hccl_socket->Connect();
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client hccl_socket Connect failed, target rank_id:" 
                         << remote_rank_info->deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", remote port:" << remoteDevicePort
                        << ", ret: " << ret;
            return -1;
        }

        hccl::HcclSocketStatus status;

        do {
            status = hccl_socket->GetStatus();
        } while (status != hccl::HcclSocketStatus::SOCKET_OK);

        ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, local_rank_info->devicePhyId, &dispatcher_);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "client HcclDispatcherInit failed, ret: " << ret;
            return -1;
        }

        notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
        if (notifyPool_ == nullptr) {
            LOG(ERROR) << "create notifyPool error";
            return -1;
        }
        ret = notifyPool_->Init(local_rank_info->devicePhyId);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "Init notifyPool error, ret: " << ret;
            return -1;
        }
        notifyPool_->RegisterOp(baseTag_);

        hccl::TransportMem::AttrInfo attrInfo;
        attrInfo.localRankId = local_rank_info->deviceLogicId;
        attrInfo.remoteRankId = remote_rank_info->deviceLogicId;
        attrInfo.sdid = 0xFFFFFFFF;
        attrInfo.serverId = local_rank_info->serverIdx;
        transport_mem = hccl::TransportMem::Create(
            hccl::TransportMem::TpType::IPC, notifyPool_, vnicNetDevCtx_, 
            dispatcher_, attrInfo);
        
        ret = transport_mem->SetDataSocket(hccl_socket);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetDataSocket failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", remote port:" << remoteDevicePort
                        << ", ret: " << ret;
            return -1;
        }
        
        ret = transport_mem->SetSocket(hccl_socket);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetSocket failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", remote port:" << remoteDevicePort
                        << ", ret: " << ret;
            return -1;
        }
        ret = transport_mem->Connect(120);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client Connect failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", remote port:" << remoteDevicePort
                        << ", ret: " << ret;
            return -1;
        }
        target_key_to_transport_mem_map_[key_str] = transport_mem;

        std::vector<hccl::TransportMem::RmaMemDesc> rmaMemDescs(g_mem_c.size());
        for (size_t i = 0; i < g_mem_c.size(); ++i) {
            LOG(INFO) << "Submit addr: " << g_mem_c[i] 
                    << " length: " << (uint64_t)g_len_c[i] 
                    << " index: " << i 
                    << " size: " << g_mem_c.size();
            hccl::TransportMem::RmaMem localRmaMem = {hccl::RmaMemType::DEVICE, g_mem_c[i], (uint64_t)g_len_c[i]};
            HCCLCHECK(transport_mem->RegMem(localRmaMem, rmaMemDescs[i]));
        }
        hccl::TransportMem::RmaMemDescs localRmaMemDescs;
        localRmaMemDescs.array = rmaMemDescs.data();
        localRmaMemDescs.arrayLength = rmaMemDescs.size();
        uint32_t actualNumOfRemote = 0;
        std::vector<hccl::TransportMem::RmaMemDesc> remoteRmaMemDescArray(g_mem_c.size());
        hccl::TransportMem::RmaMemDescs remoteRmaMemDescs;
        remoteRmaMemDescs.array = remoteRmaMemDescArray.data();
        remoteRmaMemDescs.arrayLength = g_mem_c.size();
        HCCLCHECK(transport_mem->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote));
        std::vector<hccl::TransportMem::RmaMem> remoteRmaMemArray(g_mem_c.size());
        for (uint32_t i = 0; i < g_mem_c.size(); ++i) {
            HCCLCHECK(transport_mem->EnableMemAccess(remoteRmaMemDescArray[i], remoteRmaMemArray[i]));
        }

    } else {
        transport_mem = target_key_to_transport_mem_map_[key_str];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    pid_t pid = getpid();

    hccl::TransportMem::RmaOpMem localMem;
    localMem.addr = local_mem;
    localMem.size = req_len;
    hccl::TransportMem::RmaOpMem remoteMem;
    remoteMem.addr = (void *)offset;
    remoteMem.size = req_len;

    if (op_code == WRITE){
        ret = transport_mem->Write(remoteMem, localMem, stream);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "transport_mem Write failed, localMem.addr: " 
                        << local_mem << "local_mem.size: " << req_len
                        << ", remoteMem.addr: "  << remoteMem.addr 
                        << ", remoteMem.size: " << req_len
                        << ", ret: " << ret;
            return -1;
        }
    } else {
        ret = transport_mem->Read(localMem, remoteMem, stream);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "transport_mem Read failed, localMem.addr: " 
                        << local_mem << "local_mem.size: " << req_len
                        << ", remoteMem.addr: "  << remoteMem.addr 
                        << ", remoteMem.size: " << req_len
                        << ", ret: " << ret;
            return -1;
        }
    }

    auto mid = std::chrono::high_resolution_clock::now();
    transport_mem->AddOpFence(stream);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration_sync = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto duration_call = std::chrono::duration_cast<std::chrono::microseconds>(stop - mid);
    LOG(INFO) << "pid: " << pid << "; " << "thread submit one block size: "<< req_len;
    LOG(INFO) << "pid: " << pid << "; " << "thread sync stream spent: "<< duration_sync.count() << "us";
    LOG(INFO) << "pid: " << pid << "; " << "thread call write/read spent: "<< duration_call.count() << "us";
    return 0;
}

static int acceptFromTarget(int port) {
    int client_socket;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    client_socket = accept(g_server_socket_, (struct sockaddr*)&client_addr, &client_len);
    if (client_socket < 0) {
        LOG(ERROR) << "Accept failed";
        return -1;
    }

    LOG(INFO) << "Client connected from " << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port);
    return client_socket;
}

int transportMemAccept(RankInfo *local_rank_info) {
    // 接受控制面的socket
    int result = 0;
    int client_socket = acceptFromTarget(local_rank_info->hostPort);
    if (client_socket < 0) {
        return -1;
    }
    // 接受控制面发送端的对端信息
    RankControlInfo remote_control_info;
    result = recv(client_socket, &remote_control_info, sizeof(RankControlInfo), 0);
    if (result <= 0) {
        if (result < 0) {
            LOG(ERROR) << "recv failed";
        } else {
            LOG(ERROR) << "Peer close the connection";
        }
        close(client_socket);
        return -1; // 接受对端通知消息失败
    }

    LOG(INFO) << "Received remote_control_info, deviceLogicId: "
              << remote_control_info.deviceLogicId
              << ", devicePhyId: " << remote_control_info.devicePhyId
              << ", hostIp: " << inet_ntoa(remote_control_info.hostIp);

    std::vector<unsigned int> remoteDevPhyId;
    remoteDevPhyId.push_back(remote_control_info.devicePhyId);
    HCCLCHECK(hccl::P2PMgmtPub::EnableP2P(remoteDevPhyId));
    HCCLCHECK(hccl::P2PMgmtPub::WaitP2PEnabled(remoteDevPhyId));

    hccl::HcclIpAddress remoteIp(remote_control_info.devicePhyId);
    HCCLCHECK(hrtRaGetSingleSocketVnicIpInfo(local_rank_info->devicePhyId,
                                            DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                                            remote_control_info.devicePhyId,
                                            remoteIp));

    std::vector<SocketWlistInfo> wlistInfoVec;
    SocketWlistInfo wlistInfo = {};
    wlistInfo.connLimit = 1;
    memcpy(&wlistInfo.tag[0], baseTag_.c_str(), baseTag_.size() + 1);
    wlistInfo.remoteIp.addr = remoteIp.GetBinaryAddress().addr;
    wlistInfo.remoteIp.addr6 = remoteIp.GetBinaryAddress().addr6;
    wlistInfoVec.push_back(wlistInfo);

    int ret = vnicServerSocket_->AddWhiteList(wlistInfoVec);
    if (ret != HCCL_SUCCESS) {
        LOG(ERROR) << "vnicServerSocket_ AddWhiteList failed, ret: " << ret;
        return -1;
    }
    // 接收数据面socket
    std::shared_ptr<hccl::HcclSocket> hccl_socket;
    ret = vnicServerSocket_->Accept(baseTag_, hccl_socket);
    if (ret != 0) {
        LOG(ERROR) << "transportMemAccept failed ret:" << ret;
        return -1;
    }
    // 根据host_ip+device_id，查找对应是否存在TransportMem
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    std::string key_str = inet_ntoa(remote_control_info.hostIp) + std::to_string(remote_control_info.devicePhyId);
    auto iter = target_key_to_transport_mem_map_.find(key_str);
    // 查不到，新建对应的transport_mem，可以查到，则直接复用
    if (iter == target_key_to_transport_mem_map_.end()) {
        LOG(INFO) << "The key does not exist i target_key_to_transport_mem_map, "
                "Creating transfer_mem on the accept side";
        ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, local_rank_info->devicePhyId, &dispatcher_);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "HcclDispatcherInit failed, ret: " << ret;
            return -1;
        }

        notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
        if (notifyPool_ == nullptr) {
            LOG(ERROR) << "create notifyPool error";
            return -1;
        }
        ret = notifyPool_->Init(local_rank_info->devicePhyId);
        if (ret != HCCL_SUCCESS) {
            LOG(ERROR) << "Init notifyPool error, ret: " << ret;
            return -1;
        }
        notifyPool_->RegisterOp(baseTag_);
        hccl::TransportMem::AttrInfo attrInfo;
        attrInfo.localRankId = local_rank_info->deviceLogicId;
        attrInfo.remoteRankId = remote_control_info.deviceLogicId;
        attrInfo.sdid = 0xFFFFFFFF;
        attrInfo.serverId = local_rank_info->serverIdx;
        transport_mem = hccl::TransportMem::Create(
            hccl::TransportMem::TpType::IPC, notifyPool_, vnicNetDevCtx_, 
            dispatcher_, attrInfo);

        ret = transport_mem->SetDataSocket(hccl_socket);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetDataSocket failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", ret: " << ret;
            return -1;
        }
        
        ret = transport_mem->SetSocket(hccl_socket);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetSocket failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", ret: " << ret;
            return -1;
        }
        ret = transport_mem->Connect(120);
        if (ret != HCCL_SUCCESS) {
            char deviceIp[64];
            inet_ntop(AF_INET, &remoteIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client Connect failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", remoteIp:" << deviceIp 
                        << ", ret: " << ret;
            return -1;
        }
        target_key_to_transport_mem_map_[key_str] = transport_mem;    
    } else {
        transport_mem = target_key_to_transport_mem_map_[key_str];
    }

    std::vector<hccl::TransportMem::RmaMemDesc> rmaMemDescs(g_mem_c.size());
    for (size_t i = 0; i < g_mem_c.size(); ++i) {
        hccl::TransportMem::RmaMem localRmaMem = {hccl::RmaMemType::DEVICE, g_mem_c[i], (uint64_t)g_len_c[i]};
        HCCLCHECK(transport_mem->RegMem(localRmaMem, rmaMemDescs[i]));
    }
    hccl::TransportMem::RmaMemDescs localRmaMemDescs;
    localRmaMemDescs.array = rmaMemDescs.data();
    localRmaMemDescs.arrayLength = rmaMemDescs.size();
    uint32_t actualNumOfRemote = 0;
    std::vector<hccl::TransportMem::RmaMemDesc> remoteRmaMemDescArray(g_mem_c.size());
    hccl::TransportMem::RmaMemDescs remoteRmaMemDescs;
    remoteRmaMemDescs.array = remoteRmaMemDescArray.data();
    remoteRmaMemDescs.arrayLength = g_mem_c.size();
    HCCLCHECK(transport_mem->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote));
    std::vector<hccl::TransportMem::RmaMem> remoteRmaMemArray(g_mem_c.size());
    for (uint32_t i = 0; i < g_mem_c.size(); ++i) {
        HCCLCHECK(transport_mem->EnableMemAccess(remoteRmaMemDescArray[i], remoteRmaMemArray[i]));
    }

    return 0;
}

int regLocalRmaMem(void *addr, uint64_t length)
{
    // 内存信息保存
    g_mem_c.push_back(addr);
    g_len_c.push_back(length);
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
