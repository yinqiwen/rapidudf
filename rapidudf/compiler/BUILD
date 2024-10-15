package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "function",
    hdrs = [
        "function.h",
    ],
)

cc_library(
    name = "options",
    hdrs = [
        "options.h",
    ],
)

cc_library(
    name = "type",
    srcs = [
        "type.cc",
    ],
    hdrs = [
        "type.h",
    ],
    deps = [
        "//rapidudf/log",
        "//rapidudf/meta:dtype",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
    ],
)

cc_library(
    name = "value",
    srcs = [
        "value.cc",
    ],
    hdrs = [
        "value.h",
    ],
    deps = [
        ":type",
        "//rapidudf/meta:function",
        "//rapidudf/meta:optype",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "codegen",
    srcs = [
        "codegen.cc",
        "codegen_binary.cc",
        "codegen_cast.cc",
        "codegen_ternary.cc",
        "codegen_unary.cc",
        "codegen_value.cc",
        "codegen_vector.cc",
    ],
    hdrs = [
        "codegen.h",
        "macros.h",
    ],
    deps = [
        ":options",
        ":type",
        ":value",
        "//rapidudf/functions:names",
        "//rapidudf/meta:constants",
        "//rapidudf/meta:function",
        "//rapidudf/meta:operand",
        "//rapidudf/meta:optype",
        "@local_llvm//:libllvm",
    ],
)

cc_library(
    name = "compiler",
    srcs = [
        "compiler.cc",
        "compiler_constants.cc",
        "compiler_eval.cc",
        "compiler_expressions.cc",
        "compiler_statements.cc",
    ],
    hdrs = [
        "compiler.h",
        "global_compiler.h",
    ],
    deps = [
        ":codegen",
        ":function",
        "//rapidudf/ast",
        "//rapidudf/common:lru_cache",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
