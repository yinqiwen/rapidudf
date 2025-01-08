package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "exception",
    hdrs = [
        "exception.h",
    ],
    deps = [
        "//rapidudf/log",
        "@com_github_fmtlib//:fmt",
    ],
)

cc_library(
    name = "meta",
    hdrs = [
        "type_traits.h",
    ],
    deps = [
        ":exception",
        "//rapidudf/log",
        "@com_github_fmtlib//:fmt",
    ],
)

cc_library(
    name = "optype",
    hdrs = [
        "optype.h",
    ],
)

cc_library(
    name = "operand",
    hdrs = [
        "operand.h",
    ],
    deps = [
        ":optype",
        "//rapidudf/types",
    ],
)

cc_library(
    name = "constants",
    hdrs = [
        "constants.h",
    ],
)

cc_library(
    name = "dtype_enums",
    hdrs = [
        "dtype_enums.h",
    ],
)

cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = [
        "dtype.h",
    ],
    deps = [
        ":dtype_enums",
        ":meta",
        "//rapidudf/context",
        "//rapidudf/log",
        "//rapidudf/types",
        "//rapidudf/types:vector",
        "@com_github_google_flatbuffers//:flatbuffers",
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
        ":optype",
        "//rapidudf/log",
        "//rapidudf/memory:arena",
    ],
)
