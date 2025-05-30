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

#ifndef HCCL_TRANSPORT_H
#define HCCL_TRANSPORT_H

#include <infiniband/verbs.h>

#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "transfer_metadata.h"
#include "transport/transport.h"
#include <condition_variable>

#include "hccl_network_pub.h"
#include "adapter_hccp_common.h"
#include "p2p_mgmt_pub.h"
#include "hccl_socket.h"
#include "mem_device_pub.h"
#include "dispatcher.h"
#include "notify_pool.h"
#include "transport_pub.h"
#include "dtype_common.h"
#include "hccl_ip_address.h"
#include "externalinput_pub.h"
#include "transport_mem.h"
#include "hccl_check_buf_init.h"
#include "hccl_check_common.h"
#include "hccl_opbase_rootinfo_base.h"
#include "log.h"
#include "sal_pub.h"

#define THREAD_NUM 1

namespace mooncake {
class TransferMetadata;
class HcclTransport : public Transport {
   public:
    using BufferDesc = TransferMetadata::BufferDesc;
    using SegmentDesc = TransferMetadata::SegmentDesc;
    using HandShakeDesc = TransferMetadata::HandShakeDesc;

   public:
    HcclTransport();

    ~HcclTransport();

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest> &entries) override;

    Status submitTransferTask(
        const std::vector<TransferRequest *> &request_list,
        const std::vector<TransferTask *> &task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus &status) override;

    int install(std::string &local_server_name,
                std::shared_ptr<TransferMetadata> meta, std::shared_ptr<Topology> topo) override;

    const char *getName() const override {return "hccl"; }
    
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, bool remote_accessible, bool update_metadata) override;

    int unregisterLocalMemory(void *addr, bool update_metadata = false) override;

    int registerLocalMemoryBatch(
        const std::vector<Transport::BufferEntry> &buffer_list,
        const std::string &location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void *> &addr_list) override;

   private:
    int allocateLocalSegmentID(int rank);

    void initPdThread(int rank);

    void initiatorLoop(int deviceId, int selfIdx, int rank);

    void acceptLoop(int deviceId, int selfIdx, int rank);

    int hcclInitTransportMem(int rank);

    int getRankFromServerName(const std::string& local_server_name);

   private:
    std::atomic_bool running_;
    std::thread thread_;
    std::thread allInitiatorThreads_[THREAD_NUM];
    std::thread allAcceptThreads_[THREAD_NUM];
    std::queue<Slice*> allReqQueues_[THREAD_NUM];
    std::mutex initiator_mutex_;
    std::condition_variable initiator_cond_;

    std::string baseTag_;
    std::unique_ptr<hccl::NotifyPool> notifyPool_;
    HcclNetDevCtx vnicNetDevCtx_{nullptr};
    HcclDispatcher dispatcher_;

    std::vector<std::shared_ptr<hccl::HcclSocket>> clientSocketVec_;
    std::vector<std::shared_ptr<hccl::HcclSocket>> serverSocketVec_;
    std::shared_ptr<hccl::HcclSocket> vnicServerSocket_{nullptr};
    std::unordered_map<uint64_t, std::shared_ptr<hccl::TransportMem>> target_rank_to_transport_map_;
};
}  // namespace mooncake
#endif
