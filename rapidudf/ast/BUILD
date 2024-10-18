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
        "//rapidudf/functions:names",
        "//rapidudf/log",
        "//rapidudf/meta:constants",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:operand",
        "//rapidudf/meta:optype",
        "//rapidudf/types:dyn_object_schema",
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
        "//rapidudf/functions",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "//rapidudf/types:dyn_object_schema",
        "@boost_parser//:parser",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
