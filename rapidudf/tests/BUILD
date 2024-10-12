load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "expr_bench",
    srcs = ["expr_bench.cc"],
    copts = ["-O2"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    deps = [
        "//rapidudf",
        "@com_google_benchmark//:benchmark",
        "@exprtk",
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
