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
        "//rapidudf/meta:optype",
        "//rapidudf/types",
    ],
)

cc_library(
    name = "table",
    srcs = [
        "row.cc",
        "table.cc",
        "table_schema.cc",
    ],
    hdrs = [
        "column.h",
        "row.h",
        "table.h",
        "table_schema.h",
    ],
    deps = [
        "//rapidudf/context",
        "//rapidudf/functions/simd:vector",
        "//rapidudf/reflect",
        "//rapidudf/types:dyn_object_impl",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)
