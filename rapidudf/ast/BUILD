load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = [
        "context.h",
    ],
    deps = [
        "//rapidudf/builtin:builtin_symbols",
        "//rapidudf/log",
        "//rapidudf/meta:constants",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "@com_google_absl//absl/status:statusor",
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
        "//rapidudf/builtin",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
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
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":ast",
        "@com_google_googletest//:gtest_main",
    ],
)
