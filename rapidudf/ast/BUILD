load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf:copts/configure_copts.bzl",
    "NSE_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "context",
    hdrs = [
        "context.h",
    ],
    deps = [
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:function",
        "//rapidudf/log",
    ],
)

cc_library(
    name = "ast",
    srcs = [
        "expression.cc",
        "function.cc",
        "grammar.cc",
        "statement.cc",
        "symbols.cc",
    ],
    hdrs = [
        "block.h",
        "expression.h",
        "function.h",
        "grammar.h",
        "statement.h",
        "symbols.h",
    ],
    deps = [
        ":context",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:function",
        "//rapidudf/codegen:optype",
        "//rapidudf/codegen/builtin",
        "//rapidudf/log",
        "//rapidudf/reflect",
        "@boost_parser//:parser",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grammar_test",
    size = "small",
    srcs = ["grammar_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":ast",
        "@com_google_googletest//:gtest_main",
    ],
)
