cmake_minimum_required(VERSION 3.14)
project(RapidUDF VERSION 0.1.0 LANGUAGES C CXX)
# include(FetchContent)

if( NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

option(RAPIDUDF_INSTALL "Generate the install target." ON)
option(BUILD_SHARED_LIBS
  "If enabled, build rapidudf as a shared library.  \
  This is generally discouraged, since rapidudf does not commit to having \
  a stable ABI."
  OFF
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(INCLUDE_INSTALL_DIR include CACHE STRING
  "The subdirectory where header files should be installed")
set(LIB_INSTALL_DIR lib64 CACHE STRING
  "The subdirectory where libraries should be installed")
set(BIN_INSTALL_DIR bin CACHE STRING
  "The subdirectory where binaries should be installed")
set(CMAKE_INSTALL_DIR lib64/cmake/rapidudf CACHE STRING
  "The subdirectory where CMake package config files should be installed")

include(rapidudf-deps) # Find the required packages

set(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(RAPIDUDF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rapidudf")
include(GNUInstallDirs)

list(APPEND rapidudf_srcs
${RAPIDUDF_DIR}/log/log.cc
${RAPIDUDF_DIR}/types/params.cc
${RAPIDUDF_DIR}/types/simd_vector.cc
${RAPIDUDF_DIR}/meta/dtype.cc
${RAPIDUDF_DIR}/meta/function.cc
${RAPIDUDF_DIR}/reflect/reflect.cc
${RAPIDUDF_DIR}/context/context.cc
${RAPIDUDF_DIR}/ast/context.cc
${RAPIDUDF_DIR}/ast/expression.cc
${RAPIDUDF_DIR}/ast/function.cc
${RAPIDUDF_DIR}/ast/grammar.cc
${RAPIDUDF_DIR}/ast/statement.cc
${RAPIDUDF_DIR}/ast/symbols.cc
${RAPIDUDF_DIR}/builtin/simd_vector/unary_ops.cc
${RAPIDUDF_DIR}/builtin/simd_vector/binary_ops.cc
${RAPIDUDF_DIR}/builtin/simd_vector/ternary_ops.cc
${RAPIDUDF_DIR}/builtin/simd_vector/sort_ops.cc
${RAPIDUDF_DIR}/builtin/builtin.cc
${RAPIDUDF_DIR}/builtin/json.cc
${RAPIDUDF_DIR}/builtin/stl.cc
${RAPIDUDF_DIR}/builtin/math.cc
${RAPIDUDF_DIR}/builtin/simd_vector.cc
${RAPIDUDF_DIR}/builtin/stl_maps.cc
${RAPIDUDF_DIR}/builtin/stl_sets.cc
${RAPIDUDF_DIR}/builtin/string.cc
${RAPIDUDF_DIR}/jit/llvm/unary_expression.cc
${RAPIDUDF_DIR}/jit/llvm/binary_expression.cc
${RAPIDUDF_DIR}/jit/llvm/ternary_expression.cc
${RAPIDUDF_DIR}/jit/llvm/constants.cc
${RAPIDUDF_DIR}/jit/llvm/jit_cache.cc
${RAPIDUDF_DIR}/jit/llvm/jit.cc
${RAPIDUDF_DIR}/jit/llvm/statement.cc
${RAPIDUDF_DIR}/jit/llvm/type.cc
${RAPIDUDF_DIR}/jit/llvm/value_binary_ops.cc
${RAPIDUDF_DIR}/jit/llvm/value_ternary_ops.cc
${RAPIDUDF_DIR}/jit/llvm/value_cast.cc
${RAPIDUDF_DIR}/jit/llvm/value.cc
${RAPIDUDF_DIR}/jit/llvm/var.cc
)


add_library(rapidudf_base OBJECT ${rapidudf_srcs})

if (BUILD_SHARED_LIBS)
  set_property(TARGET rapidudf_base PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()


target_include_directories(rapidudf_base PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR} 
)

target_include_directories(rapidudf_base
  PUBLIC
    $<TARGET_PROPERTY:rapidudf_deps,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(rapidudf_base
  PUBLIC
    $<TARGET_PROPERTY:rapidudf_local_deps,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(rapidudf_base
  BEFORE
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_include_directories(rapidudf_base
  INTERFACE
    $<INSTALL_INTERFACE:include>
)
add_library(rapidudf
  $<TARGET_OBJECTS:rapidudf_base>
)
target_link_libraries(rapidudf PUBLIC rapidudf_deps)
# target_link_libraries(rapidudf PRIVATE  $<TARGET_PROPERTY:rapidudf_deps,INTERFACE_LINK_LIBRARIES>)
set_target_properties(rapidudf PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib64"
)

if (RAPIDUDF_INSTALL)
set(RAPIDUDF_INSTALL_TARGETS rapidudf rapidudf_deps)
install(TARGETS ${RAPIDUDF_INSTALL_TARGETS}
  EXPORT rapidudf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/rapidudf" # source directory
  DESTINATION "include/rapidudf" # target directory
  FILES_MATCHING # install only matched files
  PATTERN "*.h" # select header files
)

  include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/rapidudf-config.cmake.in
  rapidudf-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS
    INCLUDE_INSTALL_DIR
    CMAKE_INSTALL_DIR
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rapidudf-config.cmake
  DESTINATION ${CMAKE_INSTALL_DIR}
  COMPONENT dev
)
install(
  EXPORT rapidudf
  DESTINATION ${CMAKE_INSTALL_DIR}
  NAMESPACE rapidudf::
  FILE rapidudf-targets.cmake
  COMPONENT dev
)
endif ()

