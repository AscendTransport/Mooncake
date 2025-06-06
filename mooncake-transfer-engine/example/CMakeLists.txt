link_directories("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64")
link_directories("/usr/local/mpich-3.2.1/lib/")

add_executable(transfer_engine_bench transfer_engine_bench.cpp)
target_link_libraries(transfer_engine_bench PUBLIC transfer_engine)

add_executable(memory_pool memory_pool.cpp)
target_link_libraries(memory_pool PUBLIC transfer_engine)

# add_executable(transfer_engine_ascend_test transfer_engine_ascend_test.cpp)
# target_include_directories(transfer_engine_ascend_test
#     PUBLIC
#     /usr/local/mpich-3.2.1/include/
#     /usr/local/mpich/include/
#     )
# target_link_libraries(transfer_engine_ascend_test PUBLIC transfer_engine mpi)

add_executable(transfer_engine_ascend_one_sided transfer_engine_ascend_one_sided.cpp)
target_link_libraries(transfer_engine_ascend_one_sided PUBLIC transfer_engine)