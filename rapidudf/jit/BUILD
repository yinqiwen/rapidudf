load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_COPTS",
    "RUDF_DEFAULT_LINKOPTS",
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
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        "//rapidudf/ast",
        "//rapidudf/codegen:code_generator",
        "//rapidudf/codegen/builtin",
        "//rapidudf/codegen/simd:simd_ops",
        "//rapidudf/log",
        "//rapidudf/reflect",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":jit",
        "@com_google_googletest//:gtest_main",
    ],
)
