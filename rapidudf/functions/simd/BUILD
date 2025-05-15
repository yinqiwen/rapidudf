package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "vector_op",
    hdrs = [
        "vector_op.h",
    ],
    deps = [
        "//rapidudf/meta:optype",
    ],
)

cc_library(
    name = "string",
    srcs = [
        "string.cc",
    ],
    hdrs = [
        "string.h",
    ],
    copts = ["-O3"],
    deps = [
        "//rapidudf/types",
        "@com_google_highway//:hwy",
    ],
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
        ":vector_op",
        "//rapidudf/common:allign",
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
        "@com_google_highway//:random",
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
        "bits.cc",
        "vector_binary.cc",
        "vector_ternary.cc",
        "vector_unary.cc",
    ],
    hdrs = [
        "bits.h",
        "vector.h",
    ],
    copts = [
        "-O3",
    ],
    deps = [
        ":string",
        ":vector_misc",
        ":vector_op",
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
