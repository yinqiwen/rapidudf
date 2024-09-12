load(
    "//rapidudf/copts:configure_copts.bzl",
    "RUDF_DEFAULT_COPTS",
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "function",
    hdrs = [
        "function.h",
    ],
    copts = RUDF_DEFAULT_COPTS,
    deps = [
        "//rapidudf/meta:function",
    ],
)
