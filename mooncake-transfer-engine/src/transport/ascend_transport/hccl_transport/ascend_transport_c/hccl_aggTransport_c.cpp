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

#include <iostream>
#include <unistd.h>
#include <string>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <condition_variable>
#include "transport/ascend_transport/hccl_transport/hccl_aggTransport_c.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

std::mutex g_transfer_mutex;
std::condition_variable g_transfer_cond;
std::queue<std::shared_ptr<transferReq>> g_transferReqList;

std::mutex g_split_mutex;
std::condition_variable g_split_cond;
std::queue<int> g_splitList;
std::vector<std::unique_ptr<HugeBuffer>> g_localHugeBuffer;
std::vector<uint64_t> g_localMemtoSend;

int g_hugeBufferIdx = 0;
uint64_t g_aggBlockSize = 0;

static int sendMemInfo(int client_socket, const std::vector<MemBlock> &memPool,
                       int opcode) {
    static const uint64_t kHdrLen = sizeof(opcode) + sizeof(int);
    const uint64_t kBodyLen = memPool.size() * sizeof(MemBlock);
    const uint64_t kTotal = kHdrLen + kBodyLen;

    struct iovec iov[3];
    const int mem_num = static_cast<int>(memPool.size());
    iov[0].iov_base = const_cast<void *>(static_cast<const void *>(&mem_num));
    iov[0].iov_len = sizeof(mem_num);

    iov[1].iov_base = const_cast<void *>(static_cast<const void *>(&opcode));
    iov[1].iov_len = sizeof(opcode);

    iov[2].iov_base =
        const_cast<void *>(static_cast<const void *>(memPool.data()));
    iov[2].iov_len = kBodyLen;

    uint64_t already_sent = 0;
    while (already_sent < kTotal) {
        struct msghdr msg{};
        struct iovec iov2[3];
        int iovcnt = 0;
        uint64_t skip = already_sent;

        for (int i = 0; i < 3; ++i) {
            if (skip >= iov[i].iov_len) {
                skip -= iov[i].iov_len;
                continue;
            }
            iov2[iovcnt].iov_base = static_cast<char *>(iov[i].iov_base) + skip;
            iov2[iovcnt].iov_len = iov[i].iov_len - skip;
            skip = 0;
            ++iovcnt;
        }

        msg.msg_iov = iov2;
        msg.msg_iovlen = iovcnt;

        int ret = sendmsg(client_socket, &msg, MSG_NOSIGNAL);
        if (ret < 0) {
            if (errno == EINTR) continue;
            LOG(ERROR) << "sendmsg failed: " << strerror(errno);
            return -1;
        }
        already_sent += static_cast<int>(ret);
    }

    return 0;
}

int aggTransportMemTransfer(aclrtStream stream) {
    int ret = 0;
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    std::unique_lock<std::mutex> lock(g_transfer_mutex);
    if (g_transferReqList.empty()) {
        g_transfer_cond.wait(lock);
    }
    auto req = std::move(g_transferReqList.front());
    g_transferReqList.pop();
    lock.unlock();

    hccl::TransportMem::RmaOpMem localMem;
    localMem.addr = req->local_addr;
    localMem.size = req->len;
    hccl::TransportMem::RmaOpMem remoteMem;
    remoteMem.size = req->len;
    transport_mem = g_target_key_to_connection_map[req->key_str].transport_mem;
    int opcode = req->opcode;

    int client_socket = g_target_key_to_connection_map[req->key_str].tcp_socket;
    if (req->isMerge == 0 || opcode == WRITE) {
        remoteMem.addr = req->remote_addr;
    } else if (opcode == READ) {
        uint64_t remote_base;
        ret = recv(client_socket, &remote_base, sizeof(uint64_t), MSG_WAITALL);
        if (ret <= 0) {
            LOG(ERROR) << "Failed to receive remote_base, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return -1;
        }

        remoteMem.addr = (void *)remote_base;
    }

    if (opcode == WRITE) {
        ret = transport_mem->Write(remoteMem, localMem, stream);
        if (ret) {
            LOG(ERROR) << "Write failed, localMem: " << localMem.addr
                       << ", remoteMem: " << remoteMem.addr
                       << ", req_len: " << req->len << ", ret: " << ret;
            return ret;
        }
    } else {
        ret = transport_mem->Read(localMem, remoteMem, stream);
        if (ret) {
            LOG(ERROR) << "Read failed, localMem: " << localMem.addr
                       << ", remoteMem: " << remoteMem.addr
                       << ", req_len: " << req->len << ", ret: " << ret;
            return ret;
        }
    }

    ret = transport_mem->AddOpFence(stream);
    if (ret) {
        LOG(ERROR) << "AddOpFence failed, localMem: " << localMem.addr
                   << ", remoteMem: " << remoteMem.addr
                   << ", req_len: " << req->len << ", ret: " << ret;
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret) {
        LOG(ERROR) << "aclrtSynchronizeStream failed, localMem: "
                   << localMem.addr << ", remoteMem: " << remoteMem.addr
                   << ", req_len: " << req->len << ", ret: " << ret;
        return ret;
    }

    if (req->isMerge == 0) {
        return 0;
    }

    if (opcode == WRITE) {
        int ready = 1;
        ret = send(client_socket, &ready, sizeof(int), 0);
        if (ret < 0) {
            LOG(ERROR) << "Failed to send READY to remote, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return ret;
        }

        // Check if the object at the specified index has been freed.
        ret = g_localHugeBuffer[req->mergeIdx]->freed.load(
            std::memory_order_acquire);
        if (ret) {
            LOG(ERROR) << "Error: Object at index " << req->mergeIdx
                       << " has been freed!";
            return ret;
        }

        g_localHugeBuffer[req->mergeIdx]->freed.store(
            true, std::memory_order_release);
    } else {
        // Check if the object at the specified index has been freed.
        while (!g_localHugeBuffer[req->mergeIdx]->freed.load(
            std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        g_localHugeBuffer[req->mergeIdx]->freed.store(
            false, std::memory_order_release);
        std::unique_lock<std::mutex> lock(g_split_mutex);
        g_splitList.push(req->mergeIdx);
        lock.unlock();
        g_split_cond.notify_one();
    }

    return 0;
}

int directTransfer(RankInfo *remote_rank_info, void *local_addr,
                   void *remote_addr, uint64_t len, int opcode) {
    std::string key_str = inet_ntoa(remote_rank_info->hostIp) +
                          std::to_string(remote_rank_info->devicePhyId);
    auto req = std::make_shared<transferReq>();
    req->local_addr = local_addr;
    req->remote_addr = remote_addr;
    req->len = len;
    req->opcode = opcode;
    req->isMerge = 0;
    req->key_str = key_str;

    std::unique_lock<std::mutex> lock(g_transfer_mutex);
    g_transferReqList.push(req);
    lock.unlock();
    g_transfer_cond.notify_one();

    return 0;
}

// The function can set the environment variable AGG_BLOCK_SIZE.
void setAggBlockSize() {
    const char *env_size = getenv("AGG_BLOCK_SIZE");

    if (env_size != NULL) {
        int env_value = atoi(env_size);

        if (env_value <= 2) {
            g_aggBlockSize = 2 * 1024 * 1024;
        } else if (env_value <= 4) {
            g_aggBlockSize = 4 * 1024 * 1024;
        } else if (env_value <= 6) {
            g_aggBlockSize = 6 * 1024 * 1024;
        } else {
            g_aggBlockSize = 8 * 1024 * 1024;
        }
    } else {
        g_aggBlockSize = 8 * 1024 * 1024;
    }

    g_aggBlockSize -= RESERVED_MEMORY_SIZE;
}

int aggTransportMemTask(RankInfo *local_rank_info, RankInfo *remote_rank_info,
                        std::vector<MemBlock> &local_memPool,
                        std::vector<MemBlock> &remote_memPool, int opcode,
                        aclrtStream stream) {
    int ret = 0;
    std::shared_ptr<hccl::HcclSocket> hccl_ctrl_socket;
    std::shared_ptr<hccl::HcclSocket> hccl_data_socket;
    std::shared_ptr<hccl::TransportMem> transport_mem{};
    // Check if a connection has been established with the peer, and send local
    // information to the peer
    std::string key_str = inet_ntoa(remote_rank_info->hostIp) +
                          std::to_string(remote_rank_info->devicePhyId);
    auto iter = g_target_key_to_connection_map.find(key_str);
    if (iter == g_target_key_to_connection_map.end()) {
        ret = controlInfoSend(local_rank_info, remote_rank_info);
        if (ret) {
            LOG(ERROR) << "controlInfoSend failed, ret: " << ret;
            return ret;
        }
        bool same_host =
            local_rank_info->hostIp.s_addr == remote_rank_info->hostIp.s_addr;
        // For A2 series, internal communication among 8 cards does not cross
        // HCCS, such as communication among cards 0-7
        bool same_group = (local_rank_info->devicePhyId / 8) ==
                          (remote_rank_info->devicePhyId / 8);
        bool is_cross_hccs = !(same_host && same_group);
        if (enableAscendLogging()) {
            LOG(INFO) << "hccl transport is cross_hccs: "
                      << (is_cross_hccs ? "true (cross-hccs)"
                                        : "false (same-hccs)");
        }
        ret = createClientSocket(hccl_ctrl_socket, local_rank_info,
                                 remote_rank_info, is_cross_hccs, "ctrl");
        if (ret) {
            LOG(ERROR) << "createClientSocket hccl_ctrl_socket failed, ret: "
                       << ret;
            return ret;
        }
        g_target_key_to_connection_map[key_str].hccl_ctrl_socket =
            hccl_ctrl_socket;
        ret = createClientSocket(hccl_data_socket, local_rank_info,
                                 remote_rank_info, is_cross_hccs, "data");
        if (ret) {
            LOG(ERROR) << "createClientSocket hccl_data_socket failed, ret: "
                       << ret;
            return ret;
        }
        g_target_key_to_connection_map[key_str].hccl_data_socket =
            hccl_data_socket;

        ret = createTransportMem(local_rank_info, remote_rank_info, key_str,
                                 is_cross_hccs, transport_mem);
        if (ret) {
            LOG(ERROR) << "createTransportMem failed, ret: " << ret;
            return ret;
        }
    } else {
        transport_mem = g_target_key_to_connection_map[key_str].transport_mem;
    }

    int client_socket = g_target_key_to_connection_map[key_str].tcp_socket;
    ret = sendMemInfo(client_socket, remote_memPool, opcode);
    if (ret) {
        LOG(ERROR) << "sendMemInfo failed, ret: " << ret;
        return ret;
    }

    std::vector<uint64_t> recvBuf(HUGE_BUFFER_NUM);
    int recvIdx = 0;
    if (opcode == WRITE) {
        int total = HUGE_BUFFER_NUM * sizeof(uint64_t);

        struct iovec iov[1];
        iov[0].iov_base = recvBuf.data();
        iov[0].iov_len = total;

        struct msghdr msg = {};
        msg.msg_iov = iov;
        msg.msg_iovlen = 1;

        ret = recvmsg(client_socket, &msg, MSG_WAITALL);
        if (ret != total) {
            LOG(ERROR) << "Failed to receive msg, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return -1;
        }
    }

    g_hugeBufferIdx = 0;
    uint64_t idx = 0;
    while (idx < local_memPool.size()) {
        uint64_t mergeLen = 0;
        const MemBlock &localMem = local_memPool[idx];
        if (opcode == WRITE) {
            while (!g_localHugeBuffer[g_hugeBufferIdx]->freed.load(
                std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            g_localHugeBuffer[g_hugeBufferIdx]->freed.store(
                false, std::memory_order_release);
        }

        void *mergeAddr =
            (void *)g_localHugeBuffer[g_hugeBufferIdx]->memBlock.addr;
        while (mergeLen < g_aggBlockSize && (idx < local_memPool.size())) {
            if (opcode == WRITE) {
                ret = aclrtMemcpyAsync(mergeAddr, localMem.len,
                                       (void *)localMem.addr, localMem.len,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
                if (ret != ACL_ERROR_NONE) {
                    LOG(ERROR)
                        << "Failed to merge data from device to device, ret: "
                        << ret << ", mergeAddr: " << mergeAddr
                        << ", localMem.addr: " << localMem.addr;
                    return ret;
                }
                mergeAddr = static_cast<char *>(mergeAddr) + localMem.len;
            }

            mergeLen += localMem.len;
            idx++;
        }

        auto req = std::make_shared<transferReq>();
        if (opcode == WRITE) {
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_ERROR_NONE) {
                LOG(ERROR) << "Failed to aclrtSynchronizeStream, ret: " << ret;
                return ret;
            }
            req->remote_addr = (void *)recvBuf[recvIdx];
            recvIdx = (recvIdx + 1) % HUGE_BUFFER_NUM;
        }

        req->local_addr =
            (void *)g_localHugeBuffer[g_hugeBufferIdx]->memBlock.addr;
        req->len = mergeLen;
        req->opcode = opcode;
        req->isMerge = 1;
        req->key_str = key_str;
        req->mergeIdx = g_hugeBufferIdx;

        std::unique_lock<std::mutex> lock(g_transfer_mutex);
        g_transferReqList.push(req);
        lock.unlock();
        g_transfer_cond.notify_one();
        mergeLen = 0;
        g_hugeBufferIdx =
            (g_hugeBufferIdx + 1) % HUGE_BUFFER_NUM;
    }

    if (opcode == READ) {
        idx = 0;
        while (idx < local_memPool.size()) {
            std::unique_lock<std::mutex> lock(g_split_mutex);
            if (g_splitList.empty()) {
                g_split_cond.wait(lock);
            }

            int mergeIdx = std::move(g_splitList.front());
            g_splitList.pop();
            void *mergeAddr =
                (void *)g_localHugeBuffer[mergeIdx]->memBlock.addr;
            lock.unlock();
            uint64_t mergeLen = 0;
            while (mergeLen < g_aggBlockSize && (idx < local_memPool.size())) {
                const MemBlock &localMem = local_memPool[idx];
                ret = aclrtMemcpyAsync((void *)localMem.addr, localMem.len,
                                       mergeAddr, localMem.len,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
                if (ret != ACL_ERROR_NONE) {
                    LOG(ERROR)
                        << "Failed to split data from device to device, ret: "
                        << ret;
                    return ret;
                }
                mergeAddr = static_cast<char *>(mergeAddr) + localMem.len;
                mergeLen += localMem.len;
                idx++;
            }
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_ERROR_NONE) {
                LOG(ERROR) << "Failed to aclrtSynchronizeStream, ret: " << ret;
                return ret;
            }

            mergeLen = 0;

            ret = g_localHugeBuffer[mergeIdx]->freed.load(
                std::memory_order_acquire);
            if (ret) {
                LOG(ERROR) << "Error: Object at index " << mergeIdx
                           << " has been freed!";
                return ret;
            }

            g_localHugeBuffer[mergeIdx]->freed.store(true,
                                                     std::memory_order_release);
        }
    }

    int ready = 0;
    if (opcode == WRITE) {
        ret = recv(client_socket, &ready, sizeof(int), MSG_WAITALL);
        if (ret <= 0) {
            LOG(ERROR) << "Failed to receive ready, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return -1;
        }
    }

    // auto duration_d1 =
    // std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); auto
    // duration_d2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 -
    // t2); auto duration_d3 =
    // std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3); auto
    // duration_d4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 -
    // t4);

    // LOG(INFO) << "duration_d1: " << duration_d1.count() << "us"
    //           << ", duration_d2: " << duration_d2.count() << "us"
    //           << ", duration_d3: " << duration_d3.count() << "us"
    //           << ", duration_d4: " << duration_d4.count() << "us";

    return 0;
}

static int recvMemInfo(int client_socket, aclrtStream stream) {
    int ret = 0;
    int opcode, mem_num;

    if (recv(client_socket, &mem_num, sizeof(mem_num), MSG_WAITALL) !=
        sizeof(mem_num)) {
        LOG(ERROR) << "Failed to receive mem_num, ret: " << ret
                   << ", errno: " << errno << ", error: " << strerror(errno);
        return -1;
    }

    uint64_t total_len =
        sizeof(mem_num) + sizeof(int) + mem_num * sizeof(MemBlock);

    struct iovec iov[2];
    iov[0].iov_base = &opcode;
    iov[0].iov_len = sizeof(int);

    std::vector<MemBlock> receivedMemPool;
    receivedMemPool.resize(mem_num);
    iov[1].iov_base = receivedMemPool.data();
    iov[1].iov_len = mem_num * sizeof(MemBlock);

    struct msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = 2;

    uint64_t already_received = 0;
    while (already_received < total_len - sizeof(mem_num)) {
        ret = recvmsg(client_socket, &msg, 0);
        if (ret <= 0) {
            LOG(ERROR) << "Failed to receive msg, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return ret;
        }
        already_received += ret;

        uint64_t skip = (uint64_t)ret;
        for (int i = 0; i < 2; ++i) {
            if (skip >= iov[i].iov_len) {
                skip -= iov[i].iov_len;
                iov[i].iov_len = 0;
            } else {
                iov[i].iov_base = static_cast<char *>(iov[i].iov_base) + skip;
                iov[i].iov_len -= skip;
                break;
            }
        }
    }

    setAggBlockSize();

    g_hugeBufferIdx = 0;
    if (opcode == WRITE) {
        struct iovec iov[1];
        iov[0].iov_base = static_cast<void *>(g_localMemtoSend.data());
        iov[0].iov_len = g_localMemtoSend.size() * sizeof(uint64_t);

        struct msghdr msg = {};
        msg.msg_iov = iov;
        msg.msg_iovlen = 1;

        ret = sendmsg(client_socket, &msg, 0);
        if (ret < 0) {
            LOG(ERROR) << "Failed to send msg to remote, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return ret;
        }
    }

    uint64_t idx = 0;
    int ready = 0;
    while (idx < receivedMemPool.size()) {
        uint64_t mergeLen = 0;
        void *mergeAddr =
            (void *)g_localHugeBuffer[g_hugeBufferIdx]->memBlock.addr;
        if (opcode == READ) {
            while (mergeLen < g_aggBlockSize &&
                   (idx < receivedMemPool.size())) {
                auto &block = receivedMemPool[idx];
                ret = aclrtMemcpyAsync(mergeAddr, block.len, (void *)block.addr,
                                       block.len, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       stream);
                if (ret != ACL_ERROR_NONE) {
                    LOG(ERROR)
                        << "Failed to merge data from device to device, ret: "
                        << ret;
                    return ret;
                }
                mergeAddr = static_cast<char *>(mergeAddr) + block.len;
                idx++;
                mergeLen += block.len;
            }
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_ERROR_NONE) {
                LOG(ERROR) << "Failed to aclrtSynchronizeStream, ret: " << ret;
                return ret;
            }
            uint64_t base_addr =
                g_localHugeBuffer[g_hugeBufferIdx]->memBlock.addr;
            ret = send(client_socket, &base_addr, sizeof(uint64_t), 0);
            if (ret < 0) {
                LOG(ERROR) << "Failed to send base_addr to remote, ret: " << ret
                           << ", errno: " << errno
                           << ", error: " << strerror(errno);
                return ret;
            }
        } else {
            ret = recv(client_socket, &ready, sizeof(int), MSG_WAITALL);
            if (ret <= 0) {
                LOG(ERROR) << "Failed to receive ready, ret: " << ret
                           << ", errno: " << errno
                           << ", error: " << strerror(errno);
                return ret;
            }

            while (mergeLen < g_aggBlockSize &&
                   (idx < receivedMemPool.size())) {
                auto &block = receivedMemPool[idx];
                ret = aclrtMemcpyAsync((void *)block.addr, block.len, mergeAddr,
                                       block.len, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       stream);
                if (ret != ACL_ERROR_NONE) {
                    LOG(ERROR)
                        << "Failed to split data from device to device, ret: "
                        << ret;
                    return ret;
                }
                mergeAddr = static_cast<char *>(mergeAddr) + block.len;
                idx++;
                mergeLen += block.len;
            }

            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_ERROR_NONE) {
                LOG(ERROR) << "Failed to aclrtSynchronizeStream, ret: " << ret;
                return ret;
            }
        }
        g_hugeBufferIdx =
            (g_hugeBufferIdx + 1) % HUGE_BUFFER_NUM;
    }

    if (opcode == WRITE) {
        ret = send(client_socket, &ready, sizeof(int), 0);
        if (ret < 0) {
            LOG(ERROR) << "Failed to send ready to remote, ret: " << ret
                       << ", errno: " << errno
                       << ", error: " << strerror(errno);
            return ret;
        }
    }

    // auto duration_d1 =
    // std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); auto
    // duration_d2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 -
    // t2); auto duration_d3 =
    // std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3); auto
    // duration_d4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 -
    // t4);

    // LOG(INFO) << "duration_d1: " << duration_d1.count() << "us"
    //           << ", duration_d2: " << duration_d2.count() << "us"
    //           << ", duration_d3: " << duration_d3.count() << "us"
    //           << ", duration_d4: " << duration_d4.count() << "us";

    return 1;
}

int aggTransportMemTarget(aclrtStream stream) {
    int ret = 0;
    struct epoll_event events[MAX_EVENTS];
    int nfds = epoll_wait(g_epoll_fd_agg, events, MAX_EVENTS, -1);
    if (nfds == -1) {
        if (errno == EINTR) {
            return 0;
        } else {
            LOG(ERROR) << "epoll_wait failed: " << strerror(errno);
            return -1;
        }
    }

    for (int i = 0; i < nfds; ++i) {
        if (events[i].events & EPOLLIN) {
            int fd = events[i].data.fd;
            ret = recvMemInfo(fd, stream);
            if (ret <= 0) {
                if (ret == 0) {
                    LOG(ERROR) << "Peer closed the connection on fd: " << fd;
                    epoll_ctl(g_epoll_fd_agg, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                } else {
                    LOG(ERROR) << "Failed to recvMemInfo, ret: " << ret
                               << ", errno: " << errno;
                    if (errno != EAGAIN && errno != EWOULDBLOCK) {
                        epoll_ctl(g_epoll_fd_agg, EPOLL_CTL_DEL, fd, NULL);
                        close(fd);
                    }
                }
            }
        }
    }

    return 0;
}

void aggRegLocalMem(uint64_t addr, uint64_t length, bool isAggBuffer) {
    const uint64_t alignment = 1 << 21;
    if (addr & (alignment - 1) != 0) {
        return;
    }

    MemBlock memBlock;
    memBlock.addr = addr;
    memBlock.len = length;

    g_localBuffer.emplace_back(memBlock);

    if (isAggBuffer) {
        for (uint64_t i = 0; i < HUGE_BUFFER_NUM; ++i) {
            MemBlock perHugeBuf;
            perHugeBuf.addr = addr;
            perHugeBuf.len = PER_HUGE_BUFFER_SIZE;
            g_localMemtoSend.emplace_back(addr);
            g_localHugeBuffer.emplace_back(new HugeBuffer(perHugeBuf, true));
            addr = addr + PER_HUGE_BUFFER_SIZE;
        }
    }

    return;
}

#ifdef __cplusplus
}
#endif  // __cplusplus