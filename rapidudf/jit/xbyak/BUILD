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
    name = "function",
    srcs = ["function.cc"],
    hdrs = [
        "function.h",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "register",
    srcs = ["register.cc"],
    hdrs = [
        "register.h",
    ],
    deps = [
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
    ],
)

cc_library(
    name = "reflect",
    srcs = [
        "reflect.cc",
    ],
    hdrs = [
        "reflect.h",
    ],
    deps = [
        ":code_generator",
        "//rapidudf/reflect",
    ],
)

cc_library(
    name = "code_generator",
    srcs = [
        "code_generator.cc",
        "value.cc",
    ],
    hdrs = [
        "code_generator.h",
        "value.h",
    ],
    deps = [
        ":function",
        ":register",
        "//rapidudf/builtin:builtin_symbols",
        "//rapidudf/jit/xbyak/ops:arithmetic_ops",
        "//rapidudf/jit/xbyak/ops:bits_ops",
        "//rapidudf/jit/xbyak/ops:cast",
        "//rapidudf/jit/xbyak/ops:cmp",
        "//rapidudf/jit/xbyak/ops:copy",
        "//rapidudf/jit/xbyak/ops:logic_ops",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:optype",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "jit",
    srcs = [
        "jit.cc",
        "jit_access.cc",
        "jit_constants.cc",
        "jit_expression.cc",
        "jit_statement.cc",
    ],
    hdrs = [
        "jit.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        ":code_generator",
        ":reflect",
        "//rapidudf/ast",
        "//rapidudf/builtin",
        "//rapidudf/builtin/simd_vector:simd_ops",
        "//rapidudf/jit:function",
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

cc_test(
    name = "register_test",
    size = "small",
    srcs = ["register_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":code_generator",
        ":register",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":function",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "code_generator_test",
    size = "small",
    srcs = ["code_generator_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "reflect_test",
    size = "small",
    srcs = ["reflect_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":reflect",
        "//rapidudf/jit/xbyak:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)
