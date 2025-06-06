#ifndef HCCL_TRANSPORT_MEM_H
#define HCCL_TRANSPORT_MEM_H

#include <condition_variable>
#include <glog/logging.h>
#include <functional>

#include "acl/acl.h"
#include "adapter_hccp_common.h"
#include "dispatcher.h"
#include "dtype_common.h"
#include "externalinput_pub.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_check_buf_init.h"
#include "hccl_check_common.h"
#include "hccl_ip_address.h"
#include "hccl_network_pub.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_socket.h"
#include "log.h"
#include "mem_device_pub.h"
#include "notify_pool.h"
#include "p2p_mgmt_pub.h"
#include "sal_pub.h"

#include "transport_mem.h"
#include "transport_pub.h"




#ifdef __cplusplus
extern "C" {
#endif //__cplusplus
struct RankInfo {
    uint64_t rankId = 0xFFFFFFFF; // rank id, user rank
    uint64_t serverIdx; // the Server order in ranktable (User spec)
    struct in_addr hostIp; // host IP in local server
    uint64_t hostPort;
    uint64_t deviceLogicId;
    uint64_t devicePhyId;
    DevType deviceType{DevType::DEV_TYPE_NOSOC};
    struct in_addr deviceIp;
    uint64_t devicePort;
};

struct RankControlInfo {
    int deviceLogicId;
    int devicePhyId;
    struct in_addr hostIp;
    uint64_t addr;
    uint64_t len;
};








extern int initTransportMem(RankInfo *local_rank_info);















extern int transportMemTask(RankInfo *local_rank_info, 
                            RankInfo *remote_rank_info, void *mem_addr,
                            int mem_len, int op_code, uint64_t offset,
                            uint64_t req_len, void *local_mem, aclrtStream stream);








extern int transportMemAccept(RankInfo *local_rank_info);









extern int regLocalRmaMem(void *addr, int length);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_TRANSPORT_MEM_H

