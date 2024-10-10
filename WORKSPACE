load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:rapidudf.bzl", "rapidudf_workspace")

# _LLVM_BUILD_FILE = """
# cc_library(
#     name = "x86_simd_sort",
#     srcs = glob(["lib64/libx86simdsortcpp.a"]),
#     hdrs = glob(["include/x86simdsort.h"]),
#     includes = ["include"],
#     visibility = [ "//visibility:public" ],
# )
# """

# new_local_repository(
#     name = "x86_simd_sort",
#     build_file_content = _LLVM_BUILD_FILE,
#     path = "/usr/local",
# )

rapidudf_workspace()

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# Don't use preinstalled tools to ensure builds are as hermetic as possible
rules_foreign_cc_dependencies(register_preinstalled_tools = False)

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-1e08f8e0507b6b6b1f4416a9a22cf5c28beaba93",

    # Replace the commit hash (1e08f8e0507b6b6b1f4416a9a22cf5c28beaba93) in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/1e08f8e0507b6b6b1f4416a9a22cf5c28beaba93.tar.gz",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()
