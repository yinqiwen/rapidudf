load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_COPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "llvm_jit",
    srcs = [
        "llvm_jit.cc",
    ],
    hdrs = [
        "llvm_jit.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        "//rapidudf/ast",
        "//rapidudf/log",
        "//rapidudf/reflect",
        "@com_google_absl//absl/status:statusor",
    ],
)
