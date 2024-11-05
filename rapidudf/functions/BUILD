package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "names",
    hdrs = [
        "names.h",
    ],
)

cc_library(
    name = "functions",
    srcs = [
        "functions.cc",
        "json.cc",
        "math.cc",
        "simd_table.cc",
        "simd_vector.cc",
        "stl_maps.cc",
        "stl_sets.cc",
        "stl_vector.cc",
        "strings.cc",
    ],
    hdrs = [
        "functions.h",
    ],
    deps = [
        ":names",
        "//rapidudf/functions/simd:vector",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "//rapidudf/vector:table",
        "@sleef",
    ],
)
