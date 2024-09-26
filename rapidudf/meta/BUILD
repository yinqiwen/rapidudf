load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "exception",
    hdrs = [
        "exception.h",
    ],
    deps = [
        "@com_github_fmtlib//:fmt",
    ],
)

cc_library(
    name = "meta",
    hdrs = [
        "type_traits.h",
    ],
    deps = [
        ":exception",
        "//rapidudf/arena",
        "//rapidudf/log",
        "@com_github_fmtlib//:fmt",
    ],
)

cc_library(
    name = "optype",
    hdrs = [
        "optype.h",
    ],
)

cc_library(
    name = "operand",
    hdrs = [
        "operand.h",
    ],
    deps = [
        ":optype",
        "//rapidudf/types",
    ],
)

cc_library(
    name = "constants",
    hdrs = [
        "constants.h",
    ],
)

cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = [
        "dtype.h",
    ],
    deps = [
        ":meta",
        "//rapidudf/context",
        "//rapidudf/log",
        "//rapidudf/types",
        "//rapidudf/types:scalar",
        "//rapidudf/types:simd_vector",
        "//rapidudf/types:simd_vector_table",
        "@com_github_google_flatbuffers//:runtime_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function",
    srcs = ["function.cc"],
    hdrs = [
        "function.h",
    ],
    deps = [
        ":dtype",
        ":optype",
        "//rapidudf/arena",
        "//rapidudf/log",
    ],
)

cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":dtype",
        "@com_google_googletest//:gtest_main",
    ],
)
