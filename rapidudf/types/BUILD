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
    name = "pointer",
    hdrs = [
        "pointer.h",
    ],
)

cc_library(
    name = "dyn_object",
    hdrs = [
        "dyn_object.h",
    ],
    deps = [
        ":string_view",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "dyn_object_schema",
    srcs = ["dyn_object_schema.cc"],
    hdrs = [
        "dyn_object_schema.h",
    ],
    deps = [
        ":dyn_object",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "dyn_object_impl",
    hdrs = [
        "dyn_object_impl.h",
    ],
    deps = [
        ":dyn_object",
        ":dyn_object_schema",
        "//rapidudf/log",
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
        ":dyn_object",
        ":pointer",
        ":string_view",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_nlohmann_json//:json",
    ],
)
