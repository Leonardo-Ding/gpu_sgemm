cmake_minimum_required(VERSION 2.8)
project(matrixMulDrvLoadCubin)

#find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )

link_directories( ${CUDA_LIBRARY_DIRS} )


#file(GLOB CUDA_FILES *.cu)
file(GLOB CXX_FILES *.cpp)


add_executable(
    ${PROJECT_NAME}
    ${CXX_FILES}
)

target_link_libraries(
    ${PROJECT_NAME}
    cuda
)

