package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "concurrent_clock_cache",
    srcs = ["concurrent_clock_cache_impl.cc"],
    hdrs = [
        "concurrent_clock_cache.h",
        "concurrent_clock_cache_impl.h",
    ],
    deps = [
        "//rapidudf/common:allign",
        "//rapidudf/functions/simd:vector",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
