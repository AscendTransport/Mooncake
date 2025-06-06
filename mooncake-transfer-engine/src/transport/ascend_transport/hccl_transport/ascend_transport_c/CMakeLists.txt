file(GLOB ASCEND_TRANSPORT_MEM_SOURCES "*.cpp")

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

string(REPLACE "-O2" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-O2" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

set(ASCEND_DIR "/usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha002/x86_64-linux/include")
link_directories(/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64)
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
                    /usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha002/tools/hccl_test/common/src/
                    )
add_library(ascend_transport_mem SHARED  ${ASCEND_TRANSPORT_MEM_SOURCES})
set_target_properties(ascend_transport_mem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ascend_transport_mem PRIVATE hccl ascendcl glog gflags)

target_compile_options(ascend_transport_mem BEFORE PRIVATE "-std=c++11" "-fstack-protector-strong" "-O2" "-s" "-Wl,-z,relro" "-Wl,-z,now" "-Wl,-z,noexecstack" "-Wl,--copy-dt-needed-entries" "-DOPEN_BUILD_PROJECT" "-DLOG_CPP" "-D_GLIBCXX_USE_CXX11_ABI=0")
target_link_options(ascend_transport_mem BEFORE PRIVATE "-std=c++11" "-fstack-protector-strong" "-O2" "-s" "-Wl,-z,relro" "-Wl,-z,now" "-Wl,-z,noexecstack" "-Wl,--copy-dt-needed-entries" "-DOPEN_BUILD_PROJECT" "-DLOG_CPP" "-D_GLIBCXX_USE_CXX11_ABI=0")




