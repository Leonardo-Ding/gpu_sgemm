cmake_minimum_required(VERSION 2.8)
project(matrixMul_sass)

#find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )

list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=sm_61;-keep;-std=c++11;-Xptxas -dlcm=ca;-O3")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


link_directories( ${CUDA_LIBRARY_DIRS} )
link_directories( /usr/local/cuda/lib64 )


#file(GLOB CUDA_FILES *.cu)
file(GLOB CXX_FILES *.cpp)


cuda_add_executable(
    ${PROJECT_NAME}
    ${CXX_FILES}
)

target_link_libraries(
    ${PROJECT_NAME}
    cuda
    cublas
)

