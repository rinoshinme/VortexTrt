"""
build vortex-trt library
"""
cmake_minimum_required(VERSION 2.15)

project(trt_arcface)

add_definitions(-std=c++11)

# add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
enable_language(CUDA)

if(WIN32)
	# windows
	SET(OpenCV_INCLUDE_DIRS D:/tools/opencv/build/include/)
	SET(OpenCV_LIBRARY_DIRS D:/tools/opencv/build/x64/vc15/lib/)
	SET(OpenCV_LIBS opencv_world440.lib)

	# cuda & tensorrt
	include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")
	link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/lib/x64")
	include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.2.0.6/include")
	link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.2.0.6/lib")

else()
	# unix like systems
	find_package(OpenCV 4 REQUIRED)

	# cuda & tensorrt
	include_directories(/usr/local/cuda/include)
	link_directories(/usr/local/cuda/lib64)
	include_directories(/usr/include/x86_64-linux-gnu/)
	link_directories(/usr/lib/x86_64-linux-gnu/)

	add_definitions(-O2 -pthread)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
endif(WIN32)

# project include directory
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
# project link directory
link_directories(${OpenCV_LIBRARY_DIRS})


cuda_add_executable(trt-arcface 
    arcface.cpp
    main.cpp
)

target_link_libraries(
	trt-arcface
	nvinfer
	cudart
	${OpenCV_LIBS}
)
