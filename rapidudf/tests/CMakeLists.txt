enable_testing()

# Find protobuf/flatbuffers code generation tools (use fetched versions, not system)
set(PROTOC ${CMAKE_BINARY_DIR}/_deps/protobuf-build/protoc)
set(FLATC ${CMAKE_BINARY_DIR}/_deps/flatbuffers-build/flatc)

# Generate protobuf sources (output to source dir so includes like "rapidudf/tests/test_pb.pb.h" work)
set(PROTO_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

add_custom_command(
  OUTPUT ${PROTO_OUT_DIR}/test_pb.pb.cc ${PROTO_OUT_DIR}/test_pb.pb.h
  COMMAND ${PROTOC} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --cpp_out=${PROTO_OUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_pb.proto
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_pb.proto
  COMMENT "Generating protobuf: test_pb.proto"
)
add_custom_command(
  OUTPUT ${PROTO_OUT_DIR}/pb_table_schema.pb.cc ${PROTO_OUT_DIR}/pb_table_schema.pb.h
  COMMAND ${PROTOC} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --cpp_out=${PROTO_OUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pb_table_schema.proto
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pb_table_schema.proto
  COMMENT "Generating protobuf: pb_table_schema.proto"
)

# Generate flatbuffers sources (output to source dir so includes like "rapidudf/tests/test_fbs_generated.h" work)
set(FBS_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${FBS_OUT_DIR})

add_custom_command(
  OUTPUT ${FBS_OUT_DIR}/test_fbs_generated.h
  COMMAND ${FLATC} --cpp --gen-mutable --gen-object-api --gen-compare --scoped-enums --reflect-types --reflect-names -o ${FBS_OUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_fbs.fbs
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_fbs.fbs
  COMMENT "Generating flatbuffers: test_fbs.fbs"
)
add_custom_command(
  OUTPUT ${FBS_OUT_DIR}/fbs_table_schema_generated.h
  COMMAND ${FLATC} --cpp --gen-mutable --gen-object-api --gen-compare --scoped-enums --reflect-types --reflect-names -o ${FBS_OUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/fbs_table_schema.fbs
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fbs_table_schema.fbs
  COMMENT "Generating flatbuffers: fbs_table_schema.fbs"
)

# Helper function to add a test target
function(rapidudf_add_test NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs EXTRA_SRCS EXTRA_DEPS)
  cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${NAME} ${NAME}.cc ${TEST_EXTRA_SRCS})
  target_link_libraries(${NAME} PRIVATE rapidudf GTest::gtest_main GTest::gtest)
  target_include_directories(${NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}
    ${PROTO_OUT_DIR}
    ${FBS_OUT_DIR}
  )
  if(TEST_EXTRA_DEPS)
    target_link_libraries(${NAME} PRIVATE ${TEST_EXTRA_DEPS})
  endif()
  add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

# Simple tests (depend only on rapidudf + gtest)
rapidudf_add_test(dtype_test)
rapidudf_add_test(grammar_test)
rapidudf_add_test(cmp_test)
rapidudf_add_test(logic_op_test)
rapidudf_add_test(ifelse_test)
rapidudf_add_test(while_test)
rapidudf_add_test(ternary_test)
rapidudf_add_test(cast_test)
rapidudf_add_test(math_test)
rapidudf_add_test(string_test)
rapidudf_add_test(builtin_func_test)
rapidudf_add_test(array_test)
rapidudf_add_test(simd_vector_test)
rapidudf_add_test(simd_sort_test)
rapidudf_add_test(json_test)
rapidudf_add_test(eval_engine_test)
rapidudf_add_test(arena_test)
rapidudf_add_test(dyn_object_test)
rapidudf_add_test(multi_function_test)
rapidudf_add_test(exception_test)
rapidudf_add_test(hwy_simd_test)
rapidudf_add_test(ffi_func_test)
rapidudf_add_test(ffi_struct_test)
rapidudf_add_test(ffi_stl_test)

# Tests with protobuf dependencies
rapidudf_add_test(ffi_pb_test
  EXTRA_SRCS ${PROTO_OUT_DIR}/test_pb.pb.cc
)
rapidudf_add_test(ffi_fbs_test
  EXTRA_SRCS ${FBS_OUT_DIR}/test_fbs_generated.h
)
rapidudf_add_test(simd_table_test
  EXTRA_SRCS
    ${PROTO_OUT_DIR}/pb_table_schema.pb.cc
    ${FBS_OUT_DIR}/fbs_table_schema_generated.h
    ${FBS_OUT_DIR}/test_fbs_generated.h
)

# Benchmarks (not added as tests, just as executables)
add_executable(parse_benchmark parse_benchmark.cc)
target_link_libraries(parse_benchmark PRIVATE rapidudf benchmark::benchmark)
target_include_directories(parse_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

add_executable(rapidudf_benchmark benchmark.cc)
target_link_libraries(rapidudf_benchmark PRIVATE rapidudf benchmark::benchmark)
target_include_directories(rapidudf_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

add_executable(dot_benchmark dot_benchmark.cc)
target_link_libraries(dot_benchmark PRIVATE rapidudf benchmark::benchmark)
target_include_directories(dot_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

add_executable(string_benchmark string_benchmark.cc)
target_link_libraries(string_benchmark PRIVATE rapidudf benchmark::benchmark)
target_include_directories(string_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
