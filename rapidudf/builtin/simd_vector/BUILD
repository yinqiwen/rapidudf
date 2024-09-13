load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "simd_ops",
    srcs = [
        "simd_ops.cc",
        "simd_sort.cc",
    ],
    hdrs = [
        "simd_ops.h",
    ],
    copts = [
        "-march=westmere",
        "-maes",
        "-ftree-vectorize",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/status:statusor",
        "@com_google_highway//:bit_set",
        "@com_google_highway//:dot",
        "@com_google_highway//:hwy",
        "@com_google_highway//:math",
        # "@com_google_highway//hwy/contrib/sort:vqsort",
        "@sleef",
        "@x86_simd_sort",
    ],
)

cc_test(
    name = "simd_ops_test",
    size = "small",
    srcs = ["simd_ops_test.cc"],
    copts = [
        # "-march=haswell",
        "-march=haswell",
        "-maes",
    ],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":simd_ops",
        "@com_google_googletest//:gtest_main",
        "@com_google_highway//:hwy",
    ],
)
