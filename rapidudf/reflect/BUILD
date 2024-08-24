load("@rules_cc//cc:defs.bzl", "cc_test")
load(
    "//rapidudf:copts/configure_copts.bzl",
    "NSE_DEFAULT_LINKOPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "reflect_value",
    hdrs = [
        "reflect_value.h",
    ],
)

cc_library(
    name = "reflect",
    srcs = [
        "reflect.cc",
        "stl_access.cc",
        "struct_access.cc",
    ],
    hdrs = [
        "reflect.h",
        "stl_access.h",
        "struct_access.h",
    ],
    deps = [
        ":reflect_value",
        "//rapidudf/codegen:code_generator",
        "//rapidudf/codegen:dtype",
        "//rapidudf/log",
        "@com_github_xbyak//:xbyak",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "struct_access_test",
    size = "small",
    srcs = ["struct_access_test.cc"],
    linkopts = NSE_DEFAULT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":reflect",
        "//rapidudf/codegen:code_generator",
        "@com_google_googletest//:gtest_main",
    ],
)
