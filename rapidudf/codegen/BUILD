load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf:copts/configure_copts.bzl",
    "NSE_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "optype",
    hdrs = [
        "optype.h",
    ],
)

cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = [
        "dtype.h",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/types",
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
        "//rapidudf/log",
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
        ":dtype",
        ":function",
        ":optype",
        ":register",
        "//rapidudf/codegen/builtin",
        "//rapidudf/codegen/ops:arithmetic_ops",
        "//rapidudf/codegen/ops:cast",
        "//rapidudf/codegen/ops:cmp",
        "//rapidudf/codegen/ops:copy",
        "//rapidudf/codegen/ops:logic_ops",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_test(
    name = "register_test",
    size = "small",
    srcs = ["register_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":code_generator",
        ":register",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":dtype",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
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
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)
