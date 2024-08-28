load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "copy",
    srcs = ["copy.cc"],
    hdrs = [
        "copy.h",
    ],
    deps = [
        "//rapidudf/codegen:dtype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "cast",
    srcs = ["cast.cc"],
    hdrs = [
        "cast.h",
    ],
    deps = [
        ":copy",
        "//rapidudf/codegen:dtype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "cmp",
    srcs = ["cmp.cc"],
    hdrs = [
        "cmp.h",
    ],
    deps = [
        ":copy",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:optype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "logic_ops",
    srcs = ["logic_ops.cc"],
    hdrs = [
        "logic_ops.h",
    ],
    deps = [
        ":cmp",
        ":copy",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:optype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "arithmetic_ops",
    srcs = ["arithmetic_ops.cc"],
    hdrs = [
        "arithmetic_ops.h",
    ],
    deps = [
        ":cmp",
        ":copy",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:optype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "simd_arithmetic_ops",
    srcs = ["simd_arithmetic_ops.cc"],
    hdrs = [
        "simd_arithmetic_ops.h",
    ],
    deps = [
        ":copy",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:optype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
        "@com_google_highway//:hwy",
    ],
)

cc_test(
    name = "cmp_test",
    size = "small",
    srcs = ["cmp_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":cmp",
        "//rapidudf/codegen:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cast_stack_test",
    size = "small",
    srcs = ["cast_stack_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":cast",
        "//rapidudf/codegen:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cast_register_test",
    size = "small",
    srcs = ["cast_register_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":cast",
        "//rapidudf/codegen:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)
