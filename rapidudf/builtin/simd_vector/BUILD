load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "simd_vector_ops_api",
    hdrs = [
        "column_ops.h",
        "ops.h",
        "table_ops.h",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:operand",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "//rapidudf/types:scalar",
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

cc_library(
    name = "simd_vector_ops",
    srcs = [
        "binary_ops.cc",
        "column_binary_ops.cc",
        "column_misc_ops.cc",
        "column_sort_ops.cc",
        "column_ternary_ops.cc",
        "column_unary_ops.cc",
        "eval.cc",
        "misc_ops.cc",
        "sort_ops.cc",
        "table_ops.cc",
        "ternary_ops.cc",
        "unary_ops.cc",
    ],
    copts = [
        "-O3",
        # "-ffp-contract=on",
        "-DHWY_DISABLED_TARGETS=\"(HWY_SCALAR|HWY_AVX3_SPR|HWY_AVX3_DL|HWY_AVX3_ZEN4|HWY_SSSE3|HWY_SSE4|HWY_AVX3)\"",
    ],
    deps = [
        ":simd_vector_ops_api",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:operand",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "//rapidudf/types:eval_value_impl",
        "//rapidudf/types:scalar",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
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
