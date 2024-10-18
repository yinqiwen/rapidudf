package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "vector_misc",
    srcs = [
        "vector_misc.cc",
    ],
    hdrs = [
        "vector_misc.h",
    ],
    copts = ["-O3"],
    deps = [
        "//rapidudf/context",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "@com_google_highway//:algo",
        "@com_google_highway//:bit_set",
        "@com_google_highway//:dot",
        "@com_google_highway//:hwy",
        "@com_google_highway//:math",
    ],
)

cc_library(
    name = "vector_sort",
    srcs = [
        "vector_sort.cc",
    ],
    hdrs = [
        "vector_sort.h",
    ],
    copts = ["-O3"],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "@x86_simd_sort",
    ],
)

cc_library(
    name = "vector",
    srcs = [
        "vector_binary.cc",
        "vector_ternary.cc",
        "vector_unary.cc",
    ],
    hdrs = [
        "vector.h",
    ],
    copts = ["-O3"],
    deps = [
        ":vector_misc",
        ":vector_sort",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "//rapidudf/reflect",
        "@com_google_highway//:algo",
        "@com_google_highway//:bit_set",
        "@com_google_highway//:dot",
        "@com_google_highway//:hwy",
        "@com_google_highway//:math",
        "@sleef",
    ],
)
