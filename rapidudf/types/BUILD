package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "string_view",
    hdrs = [
        "string_view.h",
    ],
    deps = [
        "@com_github_fmtlib//:fmt",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_nlohmann_json//:json",
    ],
)

cc_library(
    name = "types",
    srcs = ["params.cc"],
    hdrs = [
        "bit.h",
        "json_object.h",
        "params.h",
    ],
    deps = [
        ":string_view",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_nlohmann_json//:json",
    ],
)

cc_library(
    name = "scalar",
    srcs = ["scalar.cc"],
    hdrs = [
        "scalar.h",
    ],
    deps = [
        ":types",
        "//rapidudf/context",
        "//rapidudf/meta",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "simd_vector",
    srcs = ["simd_vector.cc"],
    hdrs = [
        "simd_vector.h",
    ],
    deps = [
        ":types",
        "//rapidudf/arena",
        "//rapidudf/meta:exception",
    ],
)

cc_library(
    name = "simd_vector_table",
    srcs = ["simd_vector_table.cc"],
    hdrs = [
        "simd_vector_table.h",
    ],
    deps = [
        ":scalar",
        ":simd_vector",
        "//rapidudf/context",
        "//rapidudf/meta:optype",
        "@com_github_fmtlib//:fmt",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)
