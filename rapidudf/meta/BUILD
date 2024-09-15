load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "meta",
    hdrs = [
        "exception.h",
        "type_traits.h",
    ],
    deps = [
        "//rapidudf/arena",
        "//rapidudf/log",
    ],
)

cc_library(
    name = "optype",
    hdrs = [
        "optype.h",
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
        "//rapidudf/types:simd_vector",
        "@com_github_google_flatbuffers//:runtime_cc",
        "@com_github_xbyak//:xbyak",
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
