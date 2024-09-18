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
    name = "type",
    srcs = [
        "type.cc",
    ],
    hdrs = [
        "type.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "jit",
    srcs = [
        "binary_expression.cc",
        "constants.cc",
        "jit.cc",
        "statement.cc",
        "ternary_expression.cc",
        "type.cc",
        "unary_expression.cc",
        "value.cc",
        "value_binary_ops.cc",
        "value_cast.cc",
        "value_ternary_ops.cc",
        "value_unary_ops.cc",
        "var.cc",
    ],
    hdrs = [
        "jit.h",
        "jit_session.h",
        "macros.h",
        "value.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        ":type",
        "//rapidudf/ast",
        "//rapidudf/jit:function",
        "//rapidudf/log",
        "//rapidudf/reflect",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "jit_cache",
    srcs = [
        "jit_cache.cc",
    ],
    hdrs = [
        "jit_cache.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        ":jit",
        "//rapidudf/common:lru_cache",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS + ["-lLLVM"],
    linkstatic = True,
    deps = [
        ":jit",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "jit_cache_test",
    size = "small",
    srcs = ["jit_cache_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS + ["-lLLVM"],
    linkstatic = True,
    deps = [
        ":jit_cache",
        "@com_google_googletest//:gtest_main",
    ],
)
