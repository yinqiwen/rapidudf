package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "vector",
    hdrs = [
        "vector.h",
    ],
    deps = [
        "//rapidudf/arena",
        "//rapidudf/meta:exception",
        "//rapidudf/types",
    ],
)

cc_library(
    name = "table",
    srcs = ["table.cc"],
    hdrs = [
        "table.h",
    ],
    deps = [
        "//rapidudf/context",
        "//rapidudf/functions/simd:vector",
        "//rapidudf/types:dyn_object_impl",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)
