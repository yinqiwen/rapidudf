# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# include(CheckCXXSourceCompiles)
# include(CheckCXXSymbolExists)
# include(CheckIncludeFileCXX)
# include(CheckFunctionExists)
# include(CMakePushCheckState)
include(FetchContent)
include(FindPkgConfig)

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
llvm_map_components_to_libnames(llvm_libs core)
list(APPEND RAPIDUDF_LINK_LIBRARIES ${llvm_libs})

find_package(Boost 1.51.0 MODULE
  COMPONENTS
    headers
  REQUIRED
)
# list(APPEND FOLLY_LINK_LIBRARIES ${Boost_LIBRARIES})
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})

find_package(PkgConfig)
pkg_check_modules(x86simdsortcpp REQUIRED IMPORTED_TARGET x86simdsortcpp)

find_package(fmt)
if(NOT fmt_FOUND)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281
  OVERRIDE_FIND_PACKAGE
) 
FetchContent_MakeAvailable(fmt)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${fmt_SOURCE_DIR}/include)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${fmt_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES fmt:fmt)

find_package(spdlog )
if(NOT spdlog_FOUND)
# set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "")
set(SPDLOG_INSTALL ON CACHE INTERNAL "")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/gabime/spdlog
  GIT_TAG        v1.14.1)
FetchContent_MakeAvailable(spdlog)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${spdlog_SOURCE_DIR}/include)
else()
# list(APPEND RAPIDUDF_LINK_LIBRARIES ${Boost_LIBRARIES})
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${spdlog_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES spdlog:spdlog)

find_package(sleef )
if(NOT sleef_FOUND)
set(SLEEF_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  sleef
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/shibatch/sleef
  GIT_TAG        3.7
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(sleef)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${sleef_BINARY_DIR}/include)
# list(APPEND RAPIDUDF_LINK_LIBRARIES sleef:sleef)
else()
# list(APPEND RAPIDUDF_LINK_LIBRARIES ${sleef_LIBRARIES})
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${sleef_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES sleef:sleef)

find_package(absl)
if(NOT absl_FOUND)
set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")
set(ABSL_ENABLE_INSTALL ON CACHE INTERNAL "")
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/abseil/abseil-cpp
  GIT_TAG        20240722.0
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(absl)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${absl_SOURCE_DIR})
else()
# list(APPEND RAPIDUDF_LINK_LIBRARIES ${absl_LIBRARIES})
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${absl_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES absl::base absl::flat_hash_map absl::utility absl::strings)

find_package(hwy)
if(NOT hwy_FOUND)
set(HWY_ENABLE_EXAMPLES OFF CACHE INTERNAL "Turn off eamples")
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  hwy
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/google/highway
  GIT_TAG        1.2.0
  FIND_PACKAGE_ARGS NAMES hwy
  # OVERRIDE_FIND_PACKAGE
)
# FetchContent_Populate(hwy)
# add_subdirectory(${hwy_SOURCE_DIR} ${hwy_BINARY_DIR})
FetchContent_MakeAvailable(hwy)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${hwy_SOURCE_DIR})

message("hwy_SOURCE_DIR: ${hwy_SOURCE_DIR}")
message("hwy_BINARY_DIR: ${hwy_BINARY_DIR}")
else()
# target_link_libraries(rapidudf_deps INTERFACE hwy::hwy hwy::hwy_contrib)
list(APPEND RAPIDUDF_LINK_LIBRARIES hwy::hwy hwy::hwy_contrib)
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${hwy_INCLUDE_DIRS})
endif()

find_package(protobuf)
if(NOT protobuf_FOUND)
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
set(protobuf_BUILD_EXAMPLES OFF CACHE INTERNAL "Turn off tests")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/protocolbuffers/protobuf
  GIT_TAG        v3.19.2
  SOURCE_SUBDIR  cmake
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(protobuf)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${protobuf_SOURCE_DIR}/src)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${protobuf_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES protobuf::libprotobuf)

find_package(flatbuffers)
if(NOT flatbuffers_FOUND)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://mirrors.tencent.com/github.com/google/flatbuffers.git
    GIT_TAG        v2.0.5 # 选择合适的版本
)
FetchContent_MakeAvailable(flatbuffers)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${flatbuffers_SOURCE_DIR}/include)
# list(APPEND RAPIDUDF_LINK_LIBRARIES flatBuffers::flatBuffers)
else()
list(APPEND RAPIDUDF_LINK_LIBRARIES flatBuffers::flatBuffers)
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${flatbuffers_INCLUDE_DIRS})
endif()


find_package(nlohmann_json)
if(NOT nlohmann_json_FOUND)
set(JSON_Install ON CACHE INTERNAL "Turn on install")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/nlohmann/json
  GIT_TAG        v3.11.3
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(nlohmann_json)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${nlohmann_json_SOURCE_DIR}/include)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${nlohmann_json_INCLUDE_DIRS})
endif()

FetchContent_Declare(
  boost_parser
  GIT_REPOSITORY https://mirrors.tencent.com/github.com/tzlaine/parser
  GIT_TAG        4cea9c03d6baf8165a21162e66be4f99ec85b529)
FetchContent_Populate(boost_parser)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${boost_parser_SOURCE_DIR}/include)

message("RAPIDUDF_LINK_LIBRARIES: ${RAPIDUDF_LINK_LIBRARIES}")
message("RAPIDUDF_INCLUDE_DIRECTORIES: ${RAPIDUDF_INCLUDE_DIRECTORIES}")

add_library(rapidudf_deps INTERFACE)
add_library(rapidudf_local_deps INTERFACE)

target_include_directories(rapidudf_deps INTERFACE ${RAPIDUDF_INCLUDE_DIRECTORIES})
target_include_directories(rapidudf_local_deps INTERFACE ${RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES})
target_link_libraries(rapidudf_deps INTERFACE
  ${RAPIDUDF_LINK_LIBRARIES}
)
