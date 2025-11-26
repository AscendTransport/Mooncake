## MindIE KV Cache池化使用指南
## 介绍
在当前大语言模型推理系统中，KV cache是广泛采用的缓存机制， Prefix Cache技术基于KV Cache技术基于KV Cache缓存机制能够在命中缓存显著减少Prefill阶段计算耗时。然而，Prefix Cache默认进使用片上显存，其容量有限，难以缓存大量前缀信息，为此，KV cache池化特性实现了存储层级的扩展，支持将更大容量的存储介质纳入前缀缓存池中，从而突破片上内存的容量限制.KV Cache池化特性能够有效提升Prefix Cache的命中率，显著降低大模型的推理成本.

## 使用介绍
KV Cache池化特性依赖于Prefix Cache特性. 此外，通过在MindIE的config.json配置文件中 `BackendConfig` 部分的一下字段配置KV Cache池化特性：
```bash
"kvPoolConfig": {"backend:":"", "configPath":""}
```
配置说明：
- `backend`: 指定使用的池化后端
- `configPath`：池化后端所需要的配置文件路径

## 已经支持的池化后端
#### Mooncake
<datails>

## Mooncake AscendTransport 编译指南
  1.下载代码分支
     git clone https://github.com/AscendTransport/Mooncake.git
     cd Mooncake
     git checkout pooling_async_memecpy_v1
  
  2.编译
     bash scripts/ascend/dependencies_ascend.sh
     cd build
     cp mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
     cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
  
  3.拷贝宿主机/usr/local/Ascend/driver/tools/到容器/usr/local/Ascend/driver/tools/
   
  4.相关依赖查看
     apt purge mpich libmpich-dev
     apt purge openmpi-bin
     apt purge openmpi-bin libopenmpi-dev
     apt install mpich libmpich-dev
     export CPATH=/usr/lib/aarch64-linux-gnu/mpich/include/:$PATH
     export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/lib:$PATH

 5.环境变量设置
    export LD_LIBRARY_PATH=/usr/lib64/mpich/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/:$LD_LIBRARY_PATH

 6.当前Mooncake AscendTransport要求`hccn.conf`，因此需要拷贝主机 `/etc/hccn.conf` 至容器内`/etc/hccn.conf`

 7.验证是否安装成功，若没有报错信息则安装成功.
```bash
mooncake_master --port 12345
```

## Mooncake AscendTransport 运行指南
使用Mooncake AscendTransport需要自行创建Mooncake Client的配置文件，可参考Mooncake Store官方配置说明和Mooncake的ascendtransport说明，创建`mooncake.json`：
```bash
{
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": "268435456",
    "protocol": "ascend",
    "device_name":"",
    "master_server_address":"master_server_ip:50001",
    "use_ascend_direct": true
}
```

## MindIE使用方式
按照前文将所有内容准备好，将Mooncake Client配置文件配置到路径 `configPath`字段， `backend`字段配置为 `mooncake`. 
1. 在终端1中拉取 `mooncake master server`：
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/:$LD_LIBRARY_PATH
mooncake_master --port 12345 --eviction_high_watermark_ratio 0.8 --eviction_ratio 0.05 --rpc_thread_num 128
```
`eviction_high_watermark_ratio`和 `eviction_ratio`属于驱逐参数，详情官方说明。 `rpc_thread_num`属于master处理client的链接并发数，建议适当提高配置用于高效处理并发请求.
2. 在终端2/3中拉取 `mooncake pd节点`：
注意环境变量设置.
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/:$LD_LIBRARY_PATH
export ASCEND_AGGREGATE_ENABLE=1
export ACL_OP_INIT_MODE=1
#A3必须配置以下环境变量，A2不需要
export ASCEND_A3_ENABLE=1
```
拉取MindIE服务化文件，具体参考MindIE社区和相关参数。

## 声明
-本代码仓提到的不池化后端仅示例，仅供您用于非商业目的。如您使用这些池化后端完成示例，请您特别注意遵守对应池化后端的Licendse，如您因使用池化后端而产生侵权纠纷，华为不承担任何责任。