load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "fib_bench",
    srcs = ["fib_bench.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    copts = ["-O2"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf/meta:dtype",
        "@com_google_googletest//:gtest_main",
    ],
)

flatbuffer_cc_library(
    name = "test_fbs",
    srcs = ["test_fbs.fbs"],
)

proto_library(
    name = "test_pb_proto",
    srcs = ["test_pb.proto"],
)

cc_proto_library(
    name = "test_pb_cc_proto",
    deps = [":test_pb_proto"],
)

cc_test(
    name = "pb_test",
    size = "small",
    srcs = ["pb_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":test_pb_cc_proto",
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "global_compiler_test",
    size = "small",
    srcs = ["global_compiler_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fbs_test",
    size = "small",
    srcs = ["fbs_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":test_fbs",
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "string_test",
    size = "small",
    srcs = ["string_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "while_test",
    size = "small",
    srcs = ["while_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ifelse_test",
    size = "small",
    srcs = ["ifelse_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logic_op_test",
    size = "small",
    srcs = ["logic_op_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "builtin_func_test",
    size = "small",
    srcs = ["builtin_func_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "exception_test",
    size = "small",
    srcs = ["exception_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "json_test",
    size = "small",
    srcs = ["json_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "multi_function_test",
    size = "small",
    srcs = ["multi_function_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "stl_test",
    size = "small",
    srcs = ["stl_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ternary_test",
    size = "small",
    srcs = ["ternary_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "struct_test",
    size = "small",
    srcs = ["struct_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simd_sort_test",
    size = "small",
    srcs = ["simd_sort_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cmp_test",
    size = "small",
    srcs = ["cmp_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "arithmetic_test",
    size = "small",
    srcs = ["arithmetic_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cast_test",
    size = "small",
    srcs = ["cast_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "math_test",
    size = "small",
    srcs = ["math_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simd_vector_test",
    size = "small",
    srcs = ["simd_vector_test.cc"],
    copts = ["-O2"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simd_table_test",
    size = "small",
    srcs = ["simd_table_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dyn_object_test",
    size = "small",
    srcs = ["dyn_object_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        "//rapidudf",
        "@com_google_googletest//:gtest_main",
    ],
)
