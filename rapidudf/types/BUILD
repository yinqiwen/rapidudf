package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    srcs = ["params.cc"],
    hdrs = [
        "bit.h",
        "json_object.h",
        "params.h",
        "string_view.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_nlohmann_json//:json",
    ],
)

cc_library(
    name = "simd",
    srcs = ["simd.cc"],
    hdrs = [
        "simd.h",
    ],
    deps = [
        ":types",
        "//rapidudf/arena",
    ],
)
