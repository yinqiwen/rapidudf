load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "student_proto",
    srcs = ["student.proto"],
)

cc_proto_library(
    name = "student_cc_proto",
    deps = [":student_proto"],
)

cc_binary(
    name = "pb_vector_table_udf",
    srcs = ["pb_vector_table_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        ":student_cc_proto",
        "//rapidudf",
    ],
)

cc_binary(
    name = "expression",
    srcs = ["expression.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)

cc_binary(
    name = "single_udf",
    srcs = ["single_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)

cc_binary(
    name = "vector_udf",
    srcs = ["vector_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)

cc_binary(
    name = "vector_table_udf",
    srcs = ["vector_table_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        ":book_fbs",
        "//rapidudf",
    ],
)

flatbuffer_cc_library(
    name = "book_fbs",
    srcs = ["book.fbs"],
    gen_reflections = True,
)

cc_binary(
    name = "fbs_vector_table_udf",
    srcs = ["fbs_vector_table_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        ":book_fbs",
        "//rapidudf",
    ],
)

cc_binary(
    name = "struct_vector_table_udf",
    srcs = ["struct_vector_table_udf.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)

cc_binary(
    name = "expression_benchmark",
    srcs = ["expression_benchmark.cc"],
    copts = [
        "-O3",
    ],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)
