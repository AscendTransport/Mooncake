cmake_minimum_required(VERSION 3.16)
project(mooncake-transfer-engine)

if (NOT GLOBAL_CONFIG)
  if (USE_ETCD)
    message(FATAL_ERROR "Cannot enable USE_ETCD while building transfer engine independently")
  endif()
  include(../mooncake-common/common.cmake)
endif() # GLOBAL_CONFIG

set(ASCEND_DIR "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include")
if (USE_ASCEND)
  include_directories(/usr/local/include
                      /usr/include
                      ${ASCEND_DIR}
                      ${ASCEND_DIR}/experiment
                      ${ASCEND_DIR}/experiment/hccl
                      ${ASCEND_DIR}/experiment/slog/toolchain
                      ${ASCEND_DIR}/experiment/metadef/common/util/error_manager
                      ${ASCEND_DIR}/experiment/runtime
                      ${ASCEND_DIR}/experiment/msprof
                      /usr/include/mpich-x86_64/
                      /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/common/src/
                      )
endif()


include_directories(include)
add_subdirectory(include)
add_subdirectory(src)

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if (WITH_RUST_EXAMPLE)
  add_subdirectory(rust)
endif()
