include(FetchContent)
include(FindPkgConfig)

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
llvm_map_components_to_libnames(llvm_libs support core target orcjit x86codegen x86asmparser)
list(APPEND RAPIDUDF_LINK_LIBRARIES ${llvm_libs})

find_package(Boost 1.51.0 REQUIRED COMPONENTS system thread)
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
list(APPEND RAPIDUDF_LINK_LIBRARIES Boost::system Boost::thread)

find_package(PkgConfig)
pkg_check_modules(x86simdsortcpp REQUIRED IMPORTED_TARGET x86simdsortcpp)
list(APPEND RAPIDUDF_LINK_LIBRARIES PkgConfig::x86simdsortcpp)


# Check system fmt headers before deciding to fetch
# (avoid find_package + FetchContent target conflict)
set(RAPIDUDF_FETCH_FMT OFF)
find_path(_FMT_CORE_H fmt/core.h HINTS /usr/local/include /usr/include)
if(_FMT_CORE_H)
  file(STRINGS "${_FMT_CORE_H}/fmt/core.h" _fmt_ver_line REGEX "^#define FMT_VERSION ")
  if(_fmt_ver_line)
    string(REGEX MATCH "[0-9]+" _fmt_hdr_ver "${_fmt_ver_line}")
    if(_fmt_hdr_ver LESS 70000)
      message(STATUS "System fmt headers version ${_fmt_hdr_ver} is too old, fetching fresh copy")
      set(RAPIDUDF_FETCH_FMT ON)
    endif()
  endif()
else()
  set(RAPIDUDF_FETCH_FMT ON)
endif()

if(RAPIDUDF_FETCH_FMT)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(fmt)
  list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${fmt_SOURCE_DIR}/include)
else()
  find_package(fmt REQUIRED)
  list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${fmt_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES fmt::fmt)

find_package(spdlog)
if(NOT spdlog_FOUND)
# set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "")
set(SPDLOG_INSTALL ON CACHE INTERNAL "")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG        v1.14.1)
FetchContent_MakeAvailable(spdlog)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${spdlog_SOURCE_DIR}/include)
else()
message("Found spdlog ${spdlog_INCLUDE_DIRS}")
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES spdlog::spdlog)

find_package(sleef)
if(NOT sleef_FOUND)
set(SLEEF_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  sleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  GIT_TAG        3.9.0
)
FetchContent_MakeAvailable(sleef)
if(TARGET sleef AND NOT TARGET sleef::sleef)
  add_library(sleef::sleef ALIAS sleef)
endif()
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${sleef_BINARY_DIR}/include)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${sleef_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES sleef::sleef)

find_package(absl)
if(NOT absl_FOUND)
set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")
set(ABSL_ENABLE_INSTALL ON CACHE INTERNAL "")
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  GIT_TAG        20250512.2
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(absl)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${absl_SOURCE_DIR})
else()
# list(APPEND RAPIDUDF_LINK_LIBRARIES ${absl_LIBRARIES})
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${absl_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES absl::base absl::cleanup absl::status absl::statusor absl::flat_hash_map  absl::flat_hash_set absl::utility absl::strings)

find_package(hwy)
if(NOT hwy_FOUND)
set(HWY_ENABLE_EXAMPLES OFF CACHE INTERNAL "Turn off eamples")
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  hwy
  GIT_REPOSITORY https://github.com/google/highway
  GIT_TAG        1.4.0
  # FIND_PACKAGE_ARGS NAMES hwy
  OVERRIDE_FIND_PACKAGE
)
# FetchContent_Populate(hwy)
# add_subdirectory(${hwy_SOURCE_DIR} ${hwy_BINARY_DIR})
FetchContent_MakeAvailable(hwy)
if(TARGET hwy AND NOT TARGET hwy::hwy)
  add_library(hwy::hwy ALIAS hwy)
endif()
if(TARGET hwy_contrib AND NOT TARGET hwy::hwy_contrib)
  add_library(hwy::hwy_contrib ALIAS hwy_contrib)
endif()
install(TARGETS hwy EXPORT rapidudfTargets)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${hwy_SOURCE_DIR})
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${hwy_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES hwy::hwy hwy::hwy_contrib)

find_package(protobuf)
if(NOT protobuf_FOUND)
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
set(protobuf_BUILD_EXAMPLES OFF CACHE INTERNAL "Turn off tests")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
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

find_package(Flatbuffers)
if(NOT Flatbuffers_FOUND)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
# GCC 13+ triggers -Werror=stringop-overflow in flatbuffers reflection.cpp
# Add -Wno-error=stringop-overflow to prevent this from being fatal
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=stringop-overflow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=stringop-overflow")
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG        v2.0.5 # 选择合适的版本
)
FetchContent_MakeAvailable(flatbuffers)
if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${flatbuffers_SOURCE_DIR}/include)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${flatbuffers_INCLUDE_DIRS})
endif()
list(APPEND RAPIDUDF_LINK_LIBRARIES flatbuffers::flatbuffers)


find_package(nlohmann_json)
if(NOT nlohmann_json_FOUND)
set(JSON_Install ON CACHE INTERNAL "Turn on install")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG        v3.11.3
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(nlohmann_json)
list(APPEND RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES ${nlohmann_json_SOURCE_DIR}/include)
else()
list(APPEND RAPIDUDF_INCLUDE_DIRECTORIES ${nlohmann_json_INCLUDE_DIRS})
endif()

# Prefer /usr/local/lib over system path to avoid header/library version mismatch
find_package(GTest HINTS /usr/local/lib/cmake/GTest /usr/local)
if(NOT GTest_FOUND)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)
endif()

find_package(benchmark)
if(NOT benchmark_FOUND)
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG        v1.8.3
)
FetchContent_MakeAvailable(benchmark)
endif()

message("RAPIDUDF_LINK_LIBRARIES: ${RAPIDUDF_LINK_LIBRARIES}")
message("RAPIDUDF_INCLUDE_DIRECTORIES: ${RAPIDUDF_INCLUDE_DIRECTORIES}")

add_library(rapidudf_deps INTERFACE)
add_library(rapidudf_local_deps INTERFACE)

target_include_directories(rapidudf_deps INTERFACE ${RAPIDUDF_INCLUDE_DIRECTORIES})
target_include_directories(rapidudf_local_deps INTERFACE ${RAPIDUDF_LOCAL_INCLUDE_DIRECTORIES})
target_link_libraries(rapidudf_deps INTERFACE
  ${RAPIDUDF_LINK_LIBRARIES}
)
