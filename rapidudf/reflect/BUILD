load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_LINKOPTS",
)

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
        "//rapidudf/codegen:code_generator",
        "//rapidudf/codegen:dtype",
        "//rapidudf/log",
        "@com_github_google_flatbuffers//:runtime_cc",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "struct_test",
    size = "small",
    srcs = ["struct_test.cc"],
    linkopts = RUDF_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":reflect",
        "//rapidudf/codegen:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)
