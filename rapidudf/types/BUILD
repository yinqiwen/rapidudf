package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "string_view",
    hdrs = [
        "string_view.h",
    ],
    deps = [
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
    name = "simd_vector",
    srcs = ["simd_vector.cc"],
    hdrs = [
        "simd_vector.h",
    ],
    deps = [
        ":types",
        "//rapidudf/arena",
    ],
)
