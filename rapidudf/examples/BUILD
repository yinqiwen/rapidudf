load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
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
    name = "simd_vector",
    srcs = ["simd_vector.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
    ],
)
