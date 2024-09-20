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
    deps = [
        ":builtin_symbols",
        "//rapidudf/builtin/simd_vector:simd_vector_ops",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "@com_github_google_flatbuffers//:runtime_cc",
    ],
)
