load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf:copts/configure_copts.bzl",
    "NSE_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "jit",
    srcs = [
        "access.cc",
        "constants.cc",
        "expression.cc",
        "jit.cc",
        "statement.cc",
    ],
    hdrs = [
        "jit.h",
    ],
    deps = [
        "//rapidudf/ast",
        "//rapidudf/codegen:code_generator",
        "//rapidudf/log",
        "//rapidudf/reflect",
    ],
)

cc_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":jit",
        "@com_google_googletest//:gtest_main",
    ],
)
