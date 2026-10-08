# ROS-free build of IMRL/FMCW-LIO. The estimator core is compiled unmodified from
# third_party/FMCW-LIO; only the ROS I/O (bithub, main) is replaced, and a tiny
# shim provides ros::Time / NodeHandle::param (YAML) / message structs.
cmake_minimum_required(VERSION 3.16)
project(fmcw_lio_standalone C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# no -march=native: AVX changes Eigen alignment and breaks ABI with the distro PCL build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_definitions(-w)

set(UP ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/FMCW-LIO)
option(BUILD_FREE_INIT "Build with Free-Init (needs Ceres)" ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB_RECURSE CORE_SRCS ${UP}/src/*.cpp ${UP}/include/*.cpp)
list(FILTER CORE_SRCS EXCLUDE REGEX ".*/main\\.cpp$")
list(FILTER CORE_SRCS EXCLUDE REGEX ".*/bithub/bithub\\.cpp$")
list(FILTER CORE_SRCS EXCLUDE REGEX ".*/free_init/.*")
# system/fmcw_lio.cpp is replaced by src/fmcw_lio_leg.cpp (adds the leg update hook)
list(FILTER CORE_SRCS EXCLUDE REGEX ".*/system/fmcw_lio\\.cpp$")
file(GLOB FREE_INIT_SRCS ${UP}/src/initializer/free_init/*.cpp)

add_executable(fmcw_lio_offline src/main_offline.cpp src/bithub_offline.cpp src/fmcw_lio_leg.cpp ${CORE_SRCS})
target_compile_definitions(fmcw_lio_offline PRIVATE ROOT_DIR="${UP}")
# order matters: our overrides (bithub, system) and the shim come before upstream
target_include_directories(fmcw_lio_offline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/shim
  ${UP}/include
  ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(fmcw_lio_offline PRIVATE ${PCL_LIBRARIES} TBB::tbb OpenMP::OpenMP_CXX yaml-cpp)
target_compile_options(fmcw_lio_offline PRIVATE -pthread -fexceptions)

if(BUILD_FREE_INIT)
  find_package(Ceres REQUIRED)
  target_compile_definitions(fmcw_lio_offline PRIVATE BUILD_FREE_INIT)
  target_sources(fmcw_lio_offline PRIVATE ${FREE_INIT_SRCS})
  target_link_libraries(fmcw_lio_offline PRIVATE Ceres::ceres)
endif()
