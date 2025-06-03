package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator",
    hdrs = [
        "allocator.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "arena",
    srcs = [
        "arena.cc",
        "folly_arena.cc",
        "leveldb_arena.cc",
    ],
    hdrs = [
        "arena.h",
        "folly_arena.h",
        "leveldb_arena.h",
    ],
    deps = [
        ":allocator",
        "//rapidudf/common:allign",
        "//rapidudf/meta:type_traits",
        "@com_github_fmtlib//:fmt",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "arena_container",
    hdrs = [
        "arena_container.h",
        "arena_string.h",
    ],
    deps = [
        ":arena",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
    ],
)
