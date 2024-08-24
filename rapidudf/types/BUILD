package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    srcs = ["params.cc"],
    hdrs = [
        "json_object.h",
        "params.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_nlohmann_json//:json",
    ],
)
