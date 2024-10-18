cmake_minimum_required(VERSION 3.11)
project(rapidudf_examples CXX)
if( NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(rapidudf REQUIRED)

add_executable(expression expression.cc)
target_link_libraries(expression PRIVATE rapidudf::rapidudf)






