#!/bin/bash

# 定义依赖路径变量
DEPENDENCIES_PATH="/mnt/deepseek/mooncake/dependencie"

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
fi

# 进入目标目录
cd "$DEPENDENCIES_PATH" || { echo "Failed to enter directory"; exit 1; }

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

pip install /mnt/deepseek/mooncake/dependencie/Mooncake/mooncake-wheel/dist/mooncake_transfer_engine-0.3.2.post1-cp311-cp311-manylinux_2_17_x86_64.whl --force-reinstall
export LD_LIBRARY_PATH=/mnt/deepseek/mooncake/dependencie/Mooncake/build/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c:$LD_LIBRARY_PATH
