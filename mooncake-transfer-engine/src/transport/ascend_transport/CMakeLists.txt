file(GLOB ASCEND_SOURCES "hccl_transport/*.cpp")

add_library(ascend_transport OBJECT ${ASCEND_SOURCES})