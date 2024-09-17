load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "simd_vector_ops",
    srcs = [
        # "simd_ops.cc",
        "unary_ops.cc",
        "binary_ops.cc",
        "ternary_ops.cc",
        "sort_ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/status:statusor",
        "@com_google_highway//:algo",
        "@com_google_highway//:bit_set",
        "@com_google_highway//:dot",
        "@com_google_highway//:hwy",
        "@com_google_highway//:math",
        "@sleef",
        "@x86_simd_sort",
    ],
)

cc_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.cc"],
    copts = [
        # "-march=haswell",
        "-march=haswell",
        "-maes",
    ],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":simd_vector_ops",
        "@com_google_googletest//:gtest_main",
        "@com_google_highway//:hwy",
    ],
)
