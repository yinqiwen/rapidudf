load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
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
