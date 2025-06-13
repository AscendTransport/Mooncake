#!/bin/bash
# 默认的git clone的依赖目录，如果docker内部git clone失败，可以把依赖的源码本地下载放到同一路径，并由TARGET_DIR指定
TARGET_DIR="/mnt/deepseek/mooncake/dependencie"

# 定义一个函数来处理git clone操作
clone_repo_if_not_exists() {
    local repo_dir=$1
    local repo_url=$2

    if [ ! -d "$repo_dir" ]; then
        git clone "$repo_url"
    else
        echo "Directory $repo_dir already exists, skipping clone."
    fi
}

set +e  # 允许脚本在某条命令失败后继续执行

# 安装基础依赖库
yum install -y cmake \
gflags-devel \
glog-devel \
libibverbs-devel \
numactl-devel \
gtest \
gtest-devel \
boost-devel \
openssl-devel --allowerasing \
hiredis-devel \
libcurl-devel \
jsoncpp-devel

# 安装 MPI 相关依赖，ASCEND依赖
yum install -y mpich mpich-devel

# 检查目录是否存在，如果不存在则执行下面的操作
if [ ! -d "/usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha002/" ]; then
    # 删除 Ascend CANN 的原有安装信息文件
    rm -rf /etc/Ascend/ascend_cann_install.info

    # 进入指定目录并安装 Ascend CANN 工具包
    /mnt/deepseek/zzy/mooncake/Ascend-cann-toolkit_8.2.RC1.alpha002_linux-x86_64.run --install --force

    # 配置 Ascend 工具包环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 更新 CPLUS_INCLUDE_PATH 环境变量，去除可能存在的旧版本 Ascend 路径
    export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | tr ':' '\n' | grep -v "/usr/local/Ascend" | paste -sd: -)
fi

# 进入目标目录
cd "$TARGET_DIR" || { echo "Failed to enter directory"; exit 1; }

# 处理 yalantinglibs
clone_repo_if_not_exists "yalantinglibs" "https://github.com/alibaba/yalantinglibs.git"
cd yalantinglibs || exit
mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j$(nproc)
make install
cd ../..

# 处理 cJSON
clone_repo_if_not_exists "cJSON" "https://github.com/DaveGamble/cJSON.git"
cd cJSON || exit
mkdir -p build
cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j
make install -j
cd ../..

# 处理 Mooncake 和 pybind11
clone_repo_if_not_exists "Mooncake" "https://github.com/AscendTransport/Mooncake.git"
clone_repo_if_not_exists "pybind11" "https://github.com/pybind/pybind11.git"

# 克隆 Mooncake 项目并进行构建和安装
cp -r pybind11/* Mooncake/extern/pybind11/
cd Mooncake

# 执行 hccl_tools.py 脚本生成本地ranktable
python scripts/hccl_tools.py

# 自动设置本地ranktable文件路径位置
export ENV_RANKTABLE_PATH=/etc/hccl_16p.json

# 创建构建目录并编译安装 Mooncake
mkdir -p build
cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j 
make install -j
cd ..

# 添加so包到环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake_vllm_adaptor.cpython-311-x86_64-linux-gnu.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/store.cpython-311-x86_64-linux-gnu.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/engine.cpython-311-x86_64-linux-gnu.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages
cp build/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages

# 复制so包到共享路径给其他人使用
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake_vllm_adaptor.cpython-311-x86_64-linux-gnu.so "$TARGET_DIR"
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/store.cpython-311-x86_64-linux-gnu.so "$TARGET_DIR"
cp /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/engine.cpython-311-x86_64-linux-gnu.so "$TARGET_DIR"
cp build/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so "$TARGET_DIR"
