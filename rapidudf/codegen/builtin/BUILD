package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "builtin",
    srcs = [
        "builtin.cc",
        "json.cc",
        "math.cc",
        "string.cc",
    ],
    hdrs = ["builtin.h"],
    deps = [
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:function",
        "//rapidudf/codegen:optype",
        "//rapidudf/codegen/simd:simd_ops",
        "//rapidudf/log",
        "@com_github_google_flatbuffers//:runtime_cc",
    ],
)
