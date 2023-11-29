cmake_minimum_required(VERSION 3.21)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

project(autd3-backend-cuda-example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AUTD_LOCAL_TEST "For local test build" OFF)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0") 
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
if(WIN32)
  FetchContent_Declare(
    autd3
    URL https://github.com/shinolab/autd3/releases/download/v17.0.2/autd3-v17.0.2-win-x64.zip
  )
else()
  FetchContent_Declare(
    autd3
    URL https://github.com/shinolab/autd3/releases/download/v17.0.2/autd3-v17.0.2-linux-x64.tar.gz
  )
endif()
FetchContent_MakeAvailable(autd3)

if(AUTD_LOCAL_TEST)
  FetchContent_Declare(
    autd3-backend-cuda
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/..
  )
elseif(WIN32)
  FetchContent_Declare(
    autd3
    URL https://github.com/shinolab/autd3-backend-cuda/releases/download/v17.0.0/autd3-backend-cuda-v17.0.0-win-x64.zip
  )
else()
  FetchContent_Declare(
    autd3
    URL https://github.com/shinolab/autd3-backend-cuda/releases/download/v17.0.0/autd3-backend-cuda-v17.0.0-linux-x64.tar.gz
  )
endif()
FetchContent_MakeAvailable(autd3-backend-cuda)

add_executable(main main.cpp)
autd3_target_link_library(main PRIVATE)
autd3_backend_cuda_target_link_library(main PRIVATE)
