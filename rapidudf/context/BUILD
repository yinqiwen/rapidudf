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
        "//rapidudf/common:allign",
        "//rapidudf/common:atomic_intrusive_list",
        "//rapidudf/log",
        "//rapidudf/memory:arena",
        "//rapidudf/meta:type_traits",
        "//rapidudf/types:vector",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)
