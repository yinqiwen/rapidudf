package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "builtin_symbols",
    hdrs = ["builtin_symbols.h"],
)

cc_library(
    name = "builtin",
    srcs = [
        "builtin.cc",
        "json.cc",
        "math.cc",
        "simd_vector.cc",
        "stl.cc",
        "stl_maps.cc",
        "stl_sets.cc",
        "string.cc",
    ],
    hdrs = ["builtin.h"],
    copts = ["-ffast-math"],
    deps = [
        ":builtin_symbols",
        "//rapidudf/codegen:dtype",
        "//rapidudf/codegen:function",
        "//rapidudf/codegen:optype",
        "//rapidudf/codegen/simd:simd_ops",
        "//rapidudf/log",
        "//rapidudf/reflect",
        "@com_github_google_flatbuffers//:runtime_cc",
    ],
)
