package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "lru_cache",
    hdrs = [
        "lru_cache.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "atomic_intrusive_list",
    hdrs = [
        "AtomicIntrusiveLinkedList.h",
    ],
)

cc_library(
    name = "allign",
    hdrs = [
        "allign.h",
    ],
)

cc_library(
    name = "variadic_template_helper",
    hdrs = [
        "variadic_template_helper.h",
    ],
)

cc_library(
    name = "arrow_helper",
    hdrs = [
        "arrow_helper.h",
    ],
    deps = [
        "//rapidudf/meta",
        "//rapidudf/types:bit",
    ],
)
