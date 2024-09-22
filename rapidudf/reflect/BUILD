package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "reflect",
    srcs = [
        "reflect.cc",
    ],
    hdrs = [
        "flatbuffers.h",
        "macros.h",
        "protobuf.h",
        "reflect.h",
        "simd_vector.h",
        "stl.h",
        "struct.h",
    ],
    deps = [
        "//rapidudf/context",
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "@com_github_google_flatbuffers//:runtime_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)
