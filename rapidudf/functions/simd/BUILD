package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "vector",
    srcs = [
        "vector_binary.cc",
        "vector_misc.cc",
        "vector_sort.cc",
        "vector_ternary.cc",
        "vector_unary.cc",
    ],
    hdrs = [
        "vector.h",
    ],
    copts = ["-O3"],
    deps = [
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
        "@x86_simd_sort",
    ],
)
