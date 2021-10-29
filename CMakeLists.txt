cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_COMPILER "dpcpp")
project (cache_determiner LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 20)
endif ()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice" )
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice" )

add_executable (${PROJECT_NAME} cache_determiner.cpp)