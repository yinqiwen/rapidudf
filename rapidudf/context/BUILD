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
        "//rapidudf/arena",
        "//rapidudf/common:atomic_intrusive_list",
        "//rapidudf/log",
        "//rapidudf/types:simd_vector",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)
