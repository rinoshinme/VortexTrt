cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)
# add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV 4 REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

SET(APP_NAME vortex)

FILE(GLOB_RECURSE CPP_SOURCE_FILES "src/*.cpp")
FILE(GLOB_RECURSE CUDA_SOURCE_FILES "src/*.cu")
FILE(GLOB_RECURSE HEADER_FILES "src/*.h")

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})


cuda_add_executable(
    ${APP_NAME}
    ${CPP_SOURCE_FILES}
    ${CUDA_SOURCE_FILES}
    ${HEADER_FILES}
)

target_link_libraries(
    ${APP_NAME}
    nvinfer
    cudart
    pthread
    ${OpenCV_LIBS}
)
