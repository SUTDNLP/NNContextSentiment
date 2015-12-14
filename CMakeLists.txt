project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
  ../LibN3L/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILdtype MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)

add_executable(WordDetector WordDetector.cpp)
add_executable(HWordDetector HWordDetector.cpp)
add_executable(HSWordDetector HSWordDetector.cpp)
add_executable(SparseDetector SparseDetector.cpp)
add_executable(RWordDetector RWordDetector.cpp)
add_executable(RHWordDetector RHWordDetector.cpp)
add_executable(RHSWordDetector RHSWordDetector.cpp)
add_executable(GRWordDetector GRWordDetector.cpp)
add_executable(GRHWordDetector GRHWordDetector.cpp)
add_executable(GRHSWordDetector GRHSWordDetector.cpp)


target_link_libraries(WordDetector openblas)
target_link_libraries(HWordDetector openblas)
target_link_libraries(HSWordDetector openblas)
target_link_libraries(SparseDetector openblas)
target_link_libraries(RWordDetector openblas)
target_link_libraries(RHWordDetector openblas)
target_link_libraries(RHSWordDetector openblas)
target_link_libraries(GRWordDetector openblas)
target_link_libraries(GRHWordDetector openblas)
target_link_libraries(GRHSWordDetector openblas)

