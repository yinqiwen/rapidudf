package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "builtin",
    srcs = [
        "builtin.cc",
        "json_funcs.cc",
        "math.cc",
        "string_funcs.cc",
    ],
    hdrs = ["builtin.h"],
    deps = [
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:function",
        "//rapidudf/codegen:optype",
        "//rapidudf/log",
        "@com_github_google_flatbuffers//:runtime_cc",
    ],
)
