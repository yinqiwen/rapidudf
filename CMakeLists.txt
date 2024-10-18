cmake_minimum_required(VERSION 3.14)
include(cmake/utils.cmake)
rapidudf_extract_version()
project(RapidUDF VERSION ${RUDF_VERSION} LANGUAGES C CXX)
message(STATUS "Build rapidudf: ${RUDF_VERSION}")

include(GNUInstallDirs)


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


include(rapidudf-deps) # Find the required packages

set(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(RAPIDUDF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rapidudf")
include(GNUInstallDirs)

file(GLOB_RECURSE rapidudf_srcs
    LIST_DIRECTORIES false 
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    CONFIGURE_DEPENDS 
    rapidudf/*.cc
)
list(FILTER rapidudf_srcs EXCLUDE REGEX "^rapidudf/tests/")
list(FILTER rapidudf_srcs EXCLUDE REGEX "^rapidudf/examples/")

foreach(SOURCE IN LISTS rapidudf_srcs)
    message(STATUS "Including source file: ${SOURCE}")
endforeach()

add_library(rapidudf_core OBJECT ${rapidudf_srcs})

# if (BUILD_SHARED_LIBS)
#   set_property(TARGET rapidudf_core PROPERTY POSITION_INDEPENDENT_CODE ON)
# endif()

target_include_directories(rapidudf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(rapidudf_core
  PUBLIC
    $<TARGET_PROPERTY:rapidudf_deps,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(rapidudf_core
  PUBLIC
    $<TARGET_PROPERTY:rapidudf_local_deps,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(rapidudf_core
  BEFORE
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
# target_include_directories(rapidudf_core
#   INTERFACE
#     $<INSTALL_INTERFACE:include>
# )
add_library(rapidudf $<TARGET_OBJECTS:rapidudf_core>)
target_link_libraries(rapidudf PUBLIC ${RAPIDUDF_LINK_LIBRARIES})
target_include_directories(rapidudf  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                                            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")


if (RAPIDUDF_INSTALL)
  set(export_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/rapidudf")
  set(project_config_in "${CMAKE_CURRENT_LIST_DIR}/cmake/rapidudfConfig.cmake.in")
  set(project_config_out "${CMAKE_CURRENT_BINARY_DIR}/rapidudfConfig.cmake")
  set(config_targets_file "rapidudfConfigTargets.cmake")
  set(version_config_file "${CMAKE_CURRENT_BINARY_DIR}/rapidudfConfigVersion.cmake")


  install(DIRECTORY "${CMAKE_SOURCE_DIR}/rapidudf" # source directory
    DESTINATION "include" # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
  )

  install(TARGETS rapidudf EXPORT rapidudfTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")


  install(EXPORT rapidudfTargets
    FILE rapidudfTargets.cmake
    NAMESPACE rapidudf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rapidudf
  )

  include(CMakePackageConfigHelpers)
  configure_package_config_file("${project_config_in}" "${project_config_out}" INSTALL_DESTINATION ${export_dest_dir})
  write_basic_package_version_file("${version_config_file}" COMPATIBILITY SameMajorVersion)
  install(FILES "${project_config_out}" "${version_config_file}" DESTINATION "${export_dest_dir}")
endif ()

