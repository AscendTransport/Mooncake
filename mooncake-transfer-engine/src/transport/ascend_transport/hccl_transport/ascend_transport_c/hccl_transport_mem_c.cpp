// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string>
#include <pthread.h>
#include <sys/time.h>
#include <errno.h>
#include "transport/ascend_transport/hccl_transport/hccl_transport_mem_c.h"

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#define READ 0
#define WRITE 1
#define CONNECT_MAX 1000 // 允许的连接数
#define RETRY_TIMES 3 //

HcclNetDevCtx vnicNetDevCtx_{nullptr};
HcclNetDevCtx nicNetDevCtx_{nullptr};

std::shared_ptr<hccl::HcclSocket> vnicServerSocket_{nullptr};
std::shared_ptr<hccl::HcclSocket> nicServerSocket_{nullptr};

std::string baseTag_ = "transport_mem";
std::unique_ptr<hccl::NotifyPool> notifyPool_;
HcclDispatcher dispatcher_{nullptr};

std::unordered_map<std::string, int>  target_key_to_control_socket_map_;
std::unordered_map<std::string, std::shared_ptr<hccl::TransportMem>> target_key_to_transport_mem_map_;
std::vector<hccl::TransportMem::RmaMem *> localRmaMem_;

std::vector<void *> g_localMemAddr;
std::vector<uint64_t> g_localMemLen;

int g_server_socket_ = 0;
struct sockaddr_in g_server_addr_;

// 初始化函数失败重试机制
#define RETRY_CALL(funcCall, errorMsg) \
    do { \
        int retryCount = 0; \
        int __ret = funcCall; \
        while (__ret && retryCount < 3) { \
            LOG(ERROR) << errorMsg << ", retrying... (" << ++retryCount << "/3)"; \
            __ret = funcCall; \
        } \
        if (__ret) { \
            LOG(ERROR) << errorMsg << " failed after 3 retries."; \
            return __ret; \
        } \
    } while (0)

static int initServerNetSocket(RankInfo *local_rank_info){
    // Init Netdev
    RETRY_CALL(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE,
    local_rank_info->devicePhyId, local_rank_info->deviceLogicId, false), "HcclNetInit failed");

    // 跨hccs使用device物理网卡IP
    hccl::HcclIpAddress localIp(local_rank_info->deviceIp);
    RETRY_CALL(HcclNetOpenDev(&nicNetDevCtx_, NicType::DEVICE_NIC_TYPE, local_rank_info->devicePhyId, 
        local_rank_info->deviceLogicId, localIp), "HcclNetOpenDev DEVICE_NIC_TYPE failed");

    nicServerSocket_ = std::make_shared<hccl::HcclSocket>(nicNetDevCtx_, local_rank_info->devicePort);
    if (nicServerSocket_ == NULL) {
        LOG(ERROR) << "make nicNetDevCtx_ failed";
        return -1;
    }

    RETRY_CALL(nicServerSocket_->Init(), "nicServerSocket_ Init failed");
    RETRY_CALL(nicServerSocket_->Listen(), "nicServerSocket_ Listen failed");

    // hccs内使用虚拟网卡
    hccl::HcclIpAddress localVnicIp(local_rank_info->devicePhyId);
    RETRY_CALL(hrtRaGetSingleSocketVnicIpInfo(
            local_rank_info->devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
            local_rank_info->devicePhyId, localVnicIp),
            "hrtRaGetSingleSocketVnicIpInfo failed");

    RETRY_CALL(HcclNetOpenDev(&vnicNetDevCtx_, NicType::VNIC_TYPE,
            local_rank_info->devicePhyId,
            local_rank_info->deviceLogicId, localVnicIp),
            "HcclNetOpenDev vnicNetDevCtx_ failed");

    // control plane connection, creat serversocket, listening client
    vnicServerSocket_ = std::make_shared<hccl::HcclSocket>(vnicNetDevCtx_, local_rank_info->devicePort);
    if (vnicServerSocket_ == NULL) {
        LOG(ERROR) << "vnicServerSocket_ make failed";
        return -1;
    }

    RETRY_CALL(vnicServerSocket_->Init(), "vnicServerSocket_ Init failed");
    RETRY_CALL(vnicServerSocket_->Listen(), "vnicServerSocket_ Listen failed");
    return 0;
}

// ascend_transport依赖的host侧带外socket，用于传递deviceId、deviceIp等控制信息
static int initControlSocket(RankInfo *local_rank_info) {
    int ret = 0;
    g_server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_socket_ < 0) {
        LOG(ERROR) << "Socket create failed";
        return g_server_socket_;
    }

    int optval = 1;
    ret = setsockopt(g_server_socket_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    if (ret < 0) {
        LOG(ERROR) << "set sock opt failed, ret: " << ret;
        close(g_server_socket_);
        return ret;
    }

    memset(&g_server_addr_, 0, sizeof(g_server_addr_));
    g_server_addr_.sin_family = AF_INET;
    g_server_addr_.sin_addr.s_addr = INADDR_ANY;
    g_server_addr_.sin_port = htons(local_rank_info->hostPort);

    ret = bind(g_server_socket_, (struct sockaddr*)&g_server_addr_, sizeof(g_server_addr_));
    if (ret < 0) {
        LOG(ERROR) << "Bind Failed, ret: " << ret;
        close(g_server_socket_);
        return ret;
    }

    // 设置接收超时
    struct timeval timeout;
    timeout.tv_sec = 120;
    timeout.tv_usec = 0;
    ret = setsockopt(g_server_socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    if (ret < 0) {
        LOG(ERROR) << "Set recv timeout failed, ret: " << ret;
        close(g_server_socket_);
        return ret;
    }

    ret = listen(g_server_socket_, CONNECT_MAX);
    if (ret < 0) {
        LOG(ERROR) << "Listen Failed, ret: " << ret;
        close(g_server_socket_);
        return ret;
    }
    LOG(INFO) << "initControlSocket successful, Server listening on host port" << ntohs(g_server_addr_.sin_port) << "..." << "g_server_socket_" << g_server_socket_;

    return 0;
}

int initTransportMem(RankInfo *local_rank_info) {
    int ret = 0;
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
    ret = initServerNetSocket(local_rank_info);
    if (ret) {
        LOG(ERROR) << "initServerNetSocket failed";
        return ret;
    }
    
    ret = initControlSocket(local_rank_info);
    if (ret) {
        LOG(ERROR) << "initControlSocket failed";
        return ret;
    }
    return 0;
}

static int connectToTarget(std::string target_ip, int target_port) {
    int client_socket;
    struct sockaddr_in server_addr;

    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        LOG(ERROR) << "Socket creation failed";
        return client_socket;
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
    
    const int max_retries = 5;
    int connected = 0;

    for (int i = 0; i < max_retries; ++i) {
        if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
            LOG(INFO) << "Connect to host server successful" << target_ip << ":" << ntohs(server_addr.sin_port);
            connected = 1;
            break;
        }

        LOG(ERROR) << "Connect attempt " << (i + 1) << " failed: " << strerror(errno);

        std::this_thread::sleep_for(std::chrono::seconds(1));  // 等待一秒再重试
    }

    if (!connected) {
        LOG(ERROR) << "Failed to connect to server after " << max_retries << " retries.";
        close(client_socket);
        return -1;
    }

    return client_socket;
}

int transportMemTask(RankInfo *local_rank_info, RankInfo *remote_rank_info, 
                    int op_code, uint64_t offset,
                    uint64_t req_len, void *local_mem, aclrtStream stream)
{
    int ret = 0;
    // 1、查找对端，检查是否具备对应的socket，并发送本端的信息给对端
    std::string key_str = inet_ntoa(remote_rank_info->hostIp) + std::to_string(remote_rank_info->devicePhyId);
    auto iter = target_key_to_control_socket_map_.find(key_str);
    if (iter == target_key_to_control_socket_map_.end()) {
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

        // 2、封装控制信息
        RankControlInfo control_info;
        control_info.deviceLogicId = local_rank_info->deviceLogicId;
        control_info.devicePhyId = local_rank_info->devicePhyId;
        control_info.hostIp = local_rank_info->hostIp;
        control_info.deviceIp = local_rank_info->deviceIp;
        // hccl_transport自建带外,发送控制面的host socket
        int client_socket = connectToTarget(inet_ntoa(remote_rank_info->hostIp), 
                                            remote_rank_info->hostPort);
        if (client_socket < 0) {
            LOG(ERROR) << "client connect failed";
            return client_socket;
        }
        ret = send(client_socket, &control_info, sizeof(RankControlInfo), 0);
        if (ret < 0) {
            LOG(ERROR) << "send control_info failed, ret: " << ret;
            close(client_socket);
            return ret;
        }
        target_key_to_control_socket_map_[key_str] = client_socket;
    }

    // 3、查找对端，检查时候具备对应的transport_mem,并发送本端的信息给对端。
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    auto iter_mem = target_key_to_transport_mem_map_.find(key_str);
    // 根据hostIp和deviceId判断是否需要跨HCCS通信，跨HCCS使用真实网卡，内部使用虚拟网卡
    bool same_host = local_rank_info->hostIp.s_addr == remote_rank_info->hostIp.s_addr;
    // 8卡内部通信不跨HCCS，如0-7卡内部通信
    bool same_group = (local_rank_info->devicePhyId / 8) == (remote_rank_info->devicePhyId / 8);
    bool is_cross_hccs = !(same_host && same_group); // 同一主机且同一组时才不跨
#ifdef ASCEND_PRINT
    LOG(INFO) << "transport is cross_hccs: " << (is_cross_hccs ? "true (cross-hccs)" : "false (same-hccs)");
#endif
    if (iter_mem == target_key_to_transport_mem_map_.end()) {
        std::shared_ptr<hccl::HcclSocket> hccl_socket;
        hccl::HcclIpAddress rempoteDevIp;
        //单机场景
        if (!is_cross_hccs) {
            std::vector<unsigned int> remoteDevPhyId;
            remoteDevPhyId.push_back(remote_rank_info->devicePhyId);
            ret = hccl::P2PMgmtPub::EnableP2P(remoteDevPhyId);
            if (ret) {
                LOG(ERROR) << "P2PMgmtPub EnableP2P faield, ret:" << ret;
                return ret;
            }
            ret = hccl::P2PMgmtPub::WaitP2PEnabled(remoteDevPhyId);
            if (ret) {
                LOG(ERROR) << "P2PMgmtPub EnableP2P faield, ret:" << ret;
                return ret;
            }
            rempoteDevIp = hccl::HcclIpAddress(remote_rank_info->devicePhyId);
            ret = hrtRaGetSingleSocketVnicIpInfo(local_rank_info->devicePhyId,
                                                DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                                                remote_rank_info->devicePhyId, rempoteDevIp);
            if (ret) {
                LOG(ERROR) << "hrtRaGetSingleSocketVnicIpInfo, ret:" << ret;
                return ret;
            }                                    
            hccl_socket = std::make_shared<hccl::HcclSocket>(
                    baseTag_, vnicNetDevCtx_, rempoteDevIp, remote_rank_info->devicePort,
                    hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
        } else {
            rempoteDevIp = hccl::HcclIpAddress(remote_rank_info->deviceIp);
            hccl_socket = 
                std::make_shared<hccl::HcclSocket>(
                    baseTag_, nicNetDevCtx_, rempoteDevIp, remote_rank_info->devicePort,
                    hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
        }

        ret = hccl_socket->Init();
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client hccl_socket init failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", remote port:" << remote_rank_info->devicePort
                        << ", ret: " << ret;
            return ret;
        }
        ret = hccl_socket->Connect();
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client hccl_socket Connect failed, target rank_id:" 
                         << remote_rank_info->deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", remote port:" << remote_rank_info->devicePort
                        << ", ret: " << ret;
            return ret;
        }

        hccl::HcclSocketStatus status;
        do {
            status = hccl_socket->GetStatus();
        } while (status != hccl::HcclSocketStatus::SOCKET_OK);
        ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, local_rank_info->devicePhyId, &dispatcher_);
        if (ret) {
            LOG(ERROR) << "client HcclDispatcherInit failed, ret: " << ret;
            return ret;
        }
        notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
        if (notifyPool_ == nullptr) {
            LOG(ERROR) << "create notifyPool error";
            return ret;
        }
        ret = notifyPool_->Init(local_rank_info->devicePhyId);
        if (ret) {
            LOG(ERROR) << "Init notifyPool error, ret: " << ret;
            return ret;
        }

        hccl::TransportMem::AttrInfo attrInfo;
        attrInfo.localRankId = local_rank_info->deviceLogicId;
        attrInfo.remoteRankId = remote_rank_info->deviceLogicId;
        attrInfo.sdid = 0xFFFFFFFF;
        attrInfo.serverId = local_rank_info->serverIdx;
        if (is_cross_hccs) {
            transport_mem = hccl::TransportMem::Create(
                hccl::TransportMem::TpType::ROCE, notifyPool_, nicNetDevCtx_, 
                dispatcher_, attrInfo);
        } else {
            transport_mem = hccl::TransportMem::Create(
                hccl::TransportMem::TpType::IPC, notifyPool_, vnicNetDevCtx_, 
                dispatcher_, attrInfo);
        }
        ret = transport_mem->SetDataSocket(hccl_socket);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetDataSocket failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", remote port:" << remote_rank_info->devicePort
                        << ", ret: " << ret;
            return ret;
        }
        ret = transport_mem->SetSocket(hccl_socket);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetSocket failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", remote port:" << remote_rank_info->devicePort
                        << ", ret: " << ret;
            return ret;
        }
        ret = transport_mem->Connect(120);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client Connect failed, target rank_id:" 
                        << remote_rank_info->deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", remote port:" << remote_rank_info->devicePort
                        << ", ret: " << ret;
            return ret;
        }
        target_key_to_transport_mem_map_[key_str] = transport_mem;
        // 适配vllm，初始化会单独注册一块host内存，2G大小，交换内存时不做处理
        size_t m_num = g_localMemAddr.size() - 1;
        std::vector<hccl::TransportMem::RmaMemDesc> rmaMemDescs(m_num);
        // 第一块内存是host内存，交换内存时不做处理
        for (size_t i = 0; i < m_num; ++i) {
            hccl::TransportMem::RmaMem localRmaMem = {hccl::RmaMemType::DEVICE, g_localMemAddr[i + 1], (uint64_t)g_localMemLen[i + 1]};
            ret = transport_mem->RegMem(localRmaMem, rmaMemDescs[i]);
            if (ret) {
                LOG(ERROR) << "transport_mem->RegMem faield, ret:" << ret << " addr: " << g_localMemAddr[i + 1] << " len: " << (uint64_t)g_localMemLen[i + 1];
                return ret;
            }
        }
        hccl::TransportMem::RmaMemDescs localRmaMemDescs;
        localRmaMemDescs.array = rmaMemDescs.data();
        localRmaMemDescs.arrayLength = rmaMemDescs.size();
        uint32_t actualNumOfRemote = 0;
        std::vector<hccl::TransportMem::RmaMemDesc> remoteRmaMemDescArray(m_num);
        hccl::TransportMem::RmaMemDescs remoteRmaMemDescs;
        remoteRmaMemDescs.array = remoteRmaMemDescArray.data();
        remoteRmaMemDescs.arrayLength = m_num;
        ret = transport_mem->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote);
        if (ret) {
            LOG(ERROR) << "transport_mem->ExchangeMemDesc faield, ret:" << ret;
            return ret;
        }
        std::vector<hccl::TransportMem::RmaMem> remoteRmaMemArray(m_num);
        for (uint32_t i = 0; i < m_num; ++i) {
            ret = transport_mem->EnableMemAccess(remoteRmaMemDescArray[i], remoteRmaMemArray[i]);
            if (ret) {
                LOG(ERROR) << "transport_mem->EnableMemAccess faield, ret:" << ret << " i:" << i;
                return ret;
            }
        }
        // 交换和使能对端内存完成，可以进行读写
        LOG(INFO) << "ExchangeMem and EnableMemAccess Success!";
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
        if (ret) {
            LOG(ERROR) << "transport_mem Write failed, localMem.addr: " 
                        << local_mem << "local_mem.size: " << req_len
                        << ", remoteMem.addr: "  << remoteMem.addr 
                        << ", remoteMem.size: " << req_len
                        << ", ret: " << ret;
            return ret;
        }
    } else {
        ret = transport_mem->Read(localMem, remoteMem, stream);
        if (ret) {
            LOG(ERROR) << "transport_mem Read failed, localMem.addr: " 
                        << local_mem << "local_mem.size: " << req_len
                        << ", remoteMem.addr: "  << remoteMem.addr 
                        << ", remoteMem.size: " << req_len
                        << ", ret: " << ret;
            return ret;
        }
    }
    auto mid = std::chrono::high_resolution_clock::now();
    ret = transport_mem->AddOpFence(stream);
    if (ret) {
        LOG(ERROR) << "transport_mem AddOpFence failed, ret: " << ret;
        return ret; 
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret) {
        LOG(ERROR) << "aclrtSynchronizeStream failed, ret: " << ret;
        return ret; 
    }

    auto stop = std::chrono::high_resolution_clock::now();
#ifdef ASCEND_PRINT
    auto duration_sync = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto duration_call = std::chrono::duration_cast<std::chrono::microseconds>(stop - mid);
    LOG(INFO) << "pid: " << pid << "; " << "thread submit one block size: "<< req_len;
    LOG(INFO) << "pid: " << pid << "; " << "thread sync stream spent: "<< duration_sync.count() << "us";
    LOG(INFO) << "pid: " << pid << "; " << "thread call write/read spent: "<< duration_call.count() << "us";
#else 
    (void)start;
    (void)mid;
    (void)stop;
    (void)pid;
#endif
    return 0;
}

static int acceptFromTarget() {
    int client_socket;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    client_socket = accept(g_server_socket_, (struct sockaddr*)&client_addr, &client_len);
    if (client_socket < 0) {
        LOG(ERROR) << "Accept failed";
        return client_socket;
    }

    LOG(INFO) << "host client connected from " << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port);
    return client_socket;
}

int transportMemAccept(RankInfo *local_rank_info) {
    // hccl_transport自建带外,接受控制面的host socket
    int ret = 0;
    int client_socket = acceptFromTarget();
    if (client_socket < 0) {
        return client_socket;
    }
    // 接受控制面发送端的对端信息
    RankControlInfo remote_control_info;
    ret = recv(client_socket, &remote_control_info, sizeof(RankControlInfo), 0);
    if (ret <= 0) {
        if (ret < 0) {
            LOG(ERROR) << "recv failed";
        } else {
            LOG(ERROR) << "Peer close the connection";
        }
        close(client_socket);
        return ret; // 接受对端通知消息失败
    }

    LOG(INFO) << "Received remote_control_info, deviceLogicId: "
              << remote_control_info.deviceLogicId
              << ", devicePhyId: " << remote_control_info.devicePhyId
              << ", hostIp: " << inet_ntoa(remote_control_info.hostIp);
    
    hccl::HcclIpAddress rempoteDevIp;
    std::shared_ptr<hccl::HcclSocket> hccl_socket;
    // 根据hostIp和deviceId判断是否需要跨HCCS通信，跨HCCS使用真实网卡，内部使用虚拟网卡
    bool same_host = local_rank_info->hostIp.s_addr == remote_control_info.hostIp.s_addr;
    // 8卡内部通信不跨HCCS，如0-7卡内部通信
    bool same_group = (local_rank_info->devicePhyId / 8) == (remote_control_info.devicePhyId / 8);
    bool is_cross_hccs = !(same_host && same_group); // 同一主机且同一组时才不跨
#ifdef ASCEND_PRINT
    LOG(INFO) << "transport is cross_hccs: " << (is_cross_hccs ? "true (cross-hccs)" : "false (same-hccs)");
#endif
    if (!is_cross_hccs) {
        std::vector<unsigned int> remoteDevPhyId;
        remoteDevPhyId.push_back(remote_control_info.devicePhyId);
        ret = hccl::P2PMgmtPub::EnableP2P(remoteDevPhyId);
        if (ret) {
            LOG(ERROR) << "P2PMgmtPub EnableP2P faield, ret:" << ret;
            return ret;
        }
        ret = hccl::P2PMgmtPub::WaitP2PEnabled(remoteDevPhyId);
        if (ret) {
            LOG(ERROR) << "P2PMgmtPub EnableP2P faield, ret:" << ret;
            return ret;
        }

        rempoteDevIp = hccl::HcclIpAddress(remote_control_info.devicePhyId);
        ret = hrtRaGetSingleSocketVnicIpInfo(local_rank_info->devicePhyId,
                                                DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                                                remote_control_info.devicePhyId,
                                                rempoteDevIp);
        if (ret) {
            LOG(ERROR) << "P2PMgmtPub EnableP2P faield, ret:" << ret;
            return ret;
        }

        std::vector<SocketWlistInfo> wlistInfoVec;
        SocketWlistInfo wlistInfo = {};
        wlistInfo.connLimit = 1;
        memcpy(&wlistInfo.tag[0], baseTag_.c_str(), baseTag_.size() + 1);
        wlistInfo.remoteIp.addr = rempoteDevIp.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = rempoteDevIp.GetBinaryAddress().addr6;
        wlistInfoVec.push_back(wlistInfo);
        // 使用device侧网卡通信之前需要添加client端地址到白名单
        LOG(INFO) << "Add the client's IP address to the whitelist.";
        ret = vnicServerSocket_->AddWhiteList(wlistInfoVec);
        if (ret) {
            LOG(ERROR) << "vnicServerSocket_ AddWhiteList failed, ret: " << ret;
            return ret;
        }
        // 接收数据面socket
        ret = vnicServerSocket_->Accept(baseTag_, hccl_socket);
        if (ret) {
            LOG(ERROR) << "vnicServerSocket_ transportMemAccept failed ret:" << ret;
            return ret;
        }
    } else {
        rempoteDevIp = hccl::HcclIpAddress(remote_control_info.deviceIp);
        std::vector<SocketWlistInfo> wlistInfoVec;
        SocketWlistInfo wlistInfo = {};
        wlistInfo.connLimit = 1;
        memcpy(&wlistInfo.tag[0], baseTag_.c_str(), baseTag_.size() + 1);
        wlistInfo.remoteIp.addr = rempoteDevIp.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = rempoteDevIp.GetBinaryAddress().addr6;
        wlistInfoVec.push_back(wlistInfo);

        int ret = nicServerSocket_->AddWhiteList(wlistInfoVec);
        if (ret) {
            LOG(ERROR) << "nicServerSocket_ AddWhiteList failed, ret: " << ret;
            return ret;
        }
        // 接收数据面socket
        ret = nicServerSocket_->Accept(baseTag_, hccl_socket);
        if (ret) {
            LOG(ERROR) << "nicServerSocket_ transportMemAccept failed ret:" << ret;
            return ret;
        }
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
        if (ret) {
            LOG(ERROR) << "HcclDispatcherInit failed, ret: " << ret;
            return ret;
        }

        notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
        if (notifyPool_ == nullptr) {
            LOG(ERROR) << "create notifyPool error";
            return -1;
        }
        ret = notifyPool_->Init(local_rank_info->devicePhyId);
        if (ret) {
            LOG(ERROR) << "Init notifyPool error, ret: " << ret;
            return ret;
        }

        hccl::TransportMem::AttrInfo attrInfo;
        attrInfo.localRankId = local_rank_info->deviceLogicId;
        attrInfo.remoteRankId = remote_control_info.deviceLogicId;
        attrInfo.sdid = 0xFFFFFFFF;
        attrInfo.serverId = local_rank_info->serverIdx;
        if (is_cross_hccs) {
            transport_mem = hccl::TransportMem::Create(
                hccl::TransportMem::TpType::ROCE, notifyPool_, nicNetDevCtx_, 
                dispatcher_, attrInfo);
        } else {
            transport_mem = hccl::TransportMem::Create(
                hccl::TransportMem::TpType::IPC, notifyPool_, vnicNetDevCtx_, 
                dispatcher_, attrInfo);
        }
        ret = transport_mem->SetDataSocket(hccl_socket);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetDataSocket failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", ret: " << ret;
            return ret;
        }
        
        ret = transport_mem->SetSocket(hccl_socket);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client SetSocket failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", ret: " << ret;
            return ret;
        }
        ret = transport_mem->Connect(120);
        if (ret) {
            char deviceIp[64];
            inet_ntop(AF_INET, &rempoteDevIp, deviceIp, sizeof(deviceIp));
            LOG(ERROR) << "client Connect failed, target rank_id:" 
                        << remote_control_info.deviceLogicId
                        << ", rempoteDevIp:" << deviceIp 
                        << ", ret: " << ret;
            return ret;
        }
        target_key_to_transport_mem_map_[key_str] = transport_mem;    
    } else {
        transport_mem = target_key_to_transport_mem_map_[key_str];
    }
    // 适配vllm，初始化会单独注册一块host内存，2G大小，交换内存时不做处理
    size_t m_num = g_localMemAddr.size() - 1;
    std::vector<hccl::TransportMem::RmaMemDesc> rmaMemDescs(m_num);
    // 第一块内存是host内存，交换内存时不做处理
    for (size_t i = 0; i < m_num; ++i) {
        hccl::TransportMem::RmaMem localRmaMem = {hccl::RmaMemType::DEVICE, g_localMemAddr[i + 1], (uint64_t)g_localMemLen[i + 1]};
        ret = transport_mem->RegMem(localRmaMem, rmaMemDescs[i]);
        if (ret) {
            LOG(ERROR) << "transport_mem->RegMem faield, ret:" << ret;
            return ret;
        }
    }
    hccl::TransportMem::RmaMemDescs localRmaMemDescs;
    localRmaMemDescs.array = rmaMemDescs.data();
    localRmaMemDescs.arrayLength = rmaMemDescs.size();
    uint32_t actualNumOfRemote = 0;
    std::vector<hccl::TransportMem::RmaMemDesc> remoteRmaMemDescArray(m_num);
    hccl::TransportMem::RmaMemDescs remoteRmaMemDescs;
    remoteRmaMemDescs.array = remoteRmaMemDescArray.data();
    remoteRmaMemDescs.arrayLength = m_num;
    ret = transport_mem->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote);
    if (ret) {
        LOG(ERROR) << "transport_mem->ExchangeMemDesc faield, ret:" << ret;
        return ret;
    }
    std::vector<hccl::TransportMem::RmaMem> remoteRmaMemArray(m_num);
    for (uint32_t i = 0; i < m_num; ++i) {
        ret = transport_mem->EnableMemAccess(remoteRmaMemDescArray[i], remoteRmaMemArray[i]);
        if (ret) {
            LOG(ERROR) << "transport_mem->EnableMemAccess faield, ret:" << ret << " i:" << i;
            return ret;
        }
    }

    // 交换和使能对端内存完成，可以进行读写
    LOG(INFO) << "ExchangeMem and EnableMemAccess Success!";
    return 0;
}

int regLocalRmaMem(void *addr, uint64_t length)
{
    // 本地内存信息保存
    g_localMemAddr.push_back(addr);
    g_localMemLen.push_back(length);
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
