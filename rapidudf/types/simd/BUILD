package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "vector",
    hdrs = [
        "vector.h",
    ],
    deps = [
        "//rapidudf/arena",
        "//rapidudf/meta:exception",
        "//rapidudf/types",
    ],
)

# cc_library(
#     name = "column",
#     srcs = ["column.cc"],
#     hdrs = [
#         "column.h",
#     ],
#     deps = [
#         ":vector",
#         "//rapidudf/context",
#         "//rapidudf/meta:dtype_enums",
#         "@com_google_absl//absl/status:statusor",
#     ],
# )

cc_library(
    name = "table",
    srcs = ["table.cc"],
    hdrs = [
        "table.h",
    ],
    deps = [
        # ":column",
        "//rapidudf/context",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)
