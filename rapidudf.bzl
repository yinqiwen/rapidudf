load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def rapidudf_workspace(path_prefix = "", tf_repo_name = "", **kwargs):
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    )
    http_archive(
        name = "rules_cc",
        sha256 = "35f2fb4ea0b3e61ad64a369de284e4fbbdcdba71836a5555abb5e194cf119509",
        strip_prefix = "rules_cc-624b5d59dfb45672d4239422fa1e3de1822ee110",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
            "https://github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
        ],
    )

    # rules_proto defines abstract rules for building Protocol Buffers.
    http_archive(
        name = "rules_proto",
        sha256 = "2490dca4f249b8a9a3ab07bd1ba6eca085aaf8e45a734af92aad0c42d9dc7aaf",
        strip_prefix = "rules_proto-218ffa7dfa5408492dc86c01ee637614f8695c45",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
        ],
    )

    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.9.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.9.0.zip",
    )

    http_archive(
        name = "rules_license",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.8/rules_license-0.0.8.tar.gz",
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.8/rules_license-0.0.8.tar.gz",
        ],
        sha256 = "241b06f3097fd186ff468832150d6cc142247dc42a32aaefb56d0099895fd229",
    )

    fbs_ver = kwargs.get("fbs_ver", "2.0.0")
    fbs_name = "flatbuffers-{ver}".format(ver = fbs_ver)
    http_archive(
        name = "com_github_google_flatbuffers",
        strip_prefix = fbs_name,
        urls = [
            "https://mirrors.tencent.com/github.com/google/flatbuffers/archive/v{ver}.tar.gz".format(ver = fbs_ver),
            "https://github.com/google/flatbuffers/archive/v{ver}.tar.gz".format(ver = fbs_ver),
        ],
    )

    protobuf_ver = kwargs.get("protobuf_ver", "3.19.2")
    protobuf_name = "protobuf-{ver}".format(ver = protobuf_ver)
    http_archive(
        name = "com_google_protobuf",
        strip_prefix = protobuf_name,
        urls = [
            "https://mirrors.tencent.com/github.com/protocolbuffers/protobuf/archive/v{ver}.tar.gz".format(ver = protobuf_ver),
            "https://github.com/protocolbuffers/protobuf/archive/v{ver}.tar.gz".format(ver = protobuf_ver),
        ],
    )

    abseil_ver = kwargs.get("abseil_ver", "20240116.2")
    abseil_name = "abseil-cpp-{ver}".format(ver = abseil_ver)
    http_archive(
        name = "com_google_absl",
        strip_prefix = abseil_name,
        urls = [
            "https://mirrors.tencent.com/github.com/abseil/abseil-cpp/archive/{ver}.tar.gz".format(ver = abseil_ver),
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/{ver}.tar.gz".format(ver = abseil_ver),
        ],
    )

    gtest_ver = kwargs.get("gtest_ver", "1.14.0")
    gtest_name = "googletest-{ver}".format(ver = gtest_ver)
    http_archive(
        name = "com_google_googletest",
        strip_prefix = gtest_name,
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/v{ver}.tar.gz".format(ver = gtest_ver),
        ],
    )

    _BOOST_PARSER_BUILD_FILE = """
cc_library(
    name = "parser",
    hdrs = glob([
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

    new_git_repository(
        name = "boost_parser",
        remote = "https://github.com/tzlaine/parser.git",
        commit = "d774edc0e638a4e5bdfb5d4f22f8a01a6f6f7983",
        build_file_content = _BOOST_PARSER_BUILD_FILE,
    )

    fbs_ver = kwargs.get("fbs_ver", "2.0.0")
    fbs_name = "flatbuffers-{ver}".format(ver = fbs_ver)
    http_archive(
        name = "com_github_google_flatbuffers",
        strip_prefix = fbs_name,
        urls = [
            "https://mirrors.tencent.com/github.com/google/flatbuffers/archive/v{ver}.tar.gz".format(ver = fbs_ver),
            "https://github.com/google/flatbuffers/archive/v{ver}.tar.gz".format(ver = fbs_ver),
        ],
    )

    json_ver = kwargs.get("json_ver", "3.11.3")
    git_repository(
        name = "com_google_nlohmann_json",
        remote = "https://github.com/nlohmann/json.git",
        tag = "v{ver}".format(ver = json_ver),
    )

    _FMTLIB_BUILD_FILE = """
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "fmt",
    generate_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFMT_TEST=OFF",
        "-DFMT_DOC=OFF",
    ],
    lib_source = ":all_srcs",
    out_lib_dir = "lib64",
    out_static_libs = [
        "libfmt.a",
    ],
    visibility = ["//visibility:public"],
)
"""
    fmtlib_ver = kwargs.get("fmtlib_ver", "11.0.2")
    fmtlib_name = "fmt-{ver}".format(ver = fmtlib_ver)
    http_archive(
        name = "com_github_fmtlib",
        strip_prefix = fmtlib_name,
        urls = [
            "https://github.com/fmtlib/fmt/archive/refs/tags/{ver}.tar.gz".format(ver = fmtlib_ver),
        ],
        build_file_content = _FMTLIB_BUILD_FILE,
    )

    _SPDLOG_BUILD_FILE = """
cc_library(
    name = "spdlog",
    hdrs = glob([
        "include/**/*.h",
    ]),
    srcs= glob([
        "src/*.cpp",
    ]),
    defines = ["SPDLOG_FMT_EXTERNAL", "SPDLOG_COMPILED_LIB"],
    includes = ["include"],
    deps = ["@com_github_fmtlib//:fmt"],
    visibility = ["//visibility:public"],
)
"""
    spdlog_ver = kwargs.get("spdlog_ver", "1.14.1")
    spdlog_name = "spdlog-{ver}".format(ver = spdlog_ver)
    http_archive(
        name = "com_github_spdlog",
        strip_prefix = spdlog_name,
        urls = [
            "https://mirrors.tencent.com/github.com/gabime/spdlog/archive/v{ver}.tar.gz".format(ver = spdlog_ver),
            "https://github.com/gabime/spdlog/archive/v{ver}.tar.gz".format(ver = spdlog_ver),
        ],
        build_file_content = _SPDLOG_BUILD_FILE,
    )

    _XBYAK_BUILD_FILE = """
cc_library(
    name = "xbyak",
    hdrs = glob([
        "**/*.h",
    ]),
    includes=["./"],
    visibility = [ "//visibility:public" ],
)
"""
    xbyak_ver = kwargs.get("xbyak_ver", "7.07")
    xbyak_name = "xbyak-{ver}".format(ver = xbyak_ver)
    http_archive(
        name = "com_github_xbyak",
        strip_prefix = xbyak_name,
        urls = [
            "https://mirrors.tencent.com/github.com/herumi/xbyak/archive/v{ver}.tar.gz".format(ver = xbyak_ver),
            "https://github.com/herumi/xbyak/archive/v{ver}.tar.gz".format(ver = xbyak_ver),
        ],
        build_file_content = _XBYAK_BUILD_FILE,
    )

    abseil_ver = kwargs.get("abseil_ver", "20240116.2")
    abseil_name = "abseil-cpp-{ver}".format(ver = abseil_ver)
    http_archive(
        name = "com_google_absl",
        strip_prefix = abseil_name,
        urls = [
            "https://mirrors.tencent.com/github.com/abseil/abseil-cpp/archive/{ver}.tar.gz".format(ver = abseil_ver),
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/{ver}.tar.gz".format(ver = abseil_ver),
        ],
    )

    git_repository(
        name = "com_google_highway",
        remote = "https://github.com/google/highway.git",
        tag = "1.2.0",
    )

    _X86_SIMD_SORT_BUILD_FILE = """
load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

make(
    name = "x86_simd_sort",
    lib_source = ":all_srcs",
    targets = ["staticlib","install"],
    args = ["-j8"],
    env={"DESTDIR":"$$INSTALLDIR"},
    out_lib_dir="usr/local/lib64",
    out_include_dir="usr/local/include",
    out_static_libs = ["libx86simdsortcpp.a"],
    visibility = ["//visibility:public"],
)
"""
    new_git_repository(
        name = "x86_simd_sort",
        remote = "https://github.com/intel/x86-simd-sort.git",
        commit = "c61ce8f97f495cab7077774741be7a3903e71dd4",
        build_file_content = _X86_SIMD_SORT_BUILD_FILE,
    )

    _SLEEF_BUILD_FILE = """
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "sleef",
    generate_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSLEEF_BUILD_TESTS=OFF",
    ],
    lib_source = ":all_srcs",
    out_lib_dir = "lib64",
    out_static_libs = [
        "libsleef.a",
        "libsleefgnuabi.a",
    ],
    visibility = ["//visibility:public"],
)
"""

    new_git_repository(
        name = "sleef",
        remote = "https://github.com/shibatch/sleef.git",
        tag = "3.7",
        build_file_content = _SLEEF_BUILD_FILE,
    )

    _EXPRTK_BUILD_FILE = """
cc_library(
    name = "exprtk",
    hdrs = [
        "exprtk.hpp",
    ],
    visibility = ["//visibility:public"],
)
"""
    new_git_repository(
        name = "exprtk",
        remote = "https://github.com/ArashPartow/exprtk.git",
        branch = "master",
        build_file_content = _EXPRTK_BUILD_FILE,
    )

    bench_ver = kwargs.get("bench_ver", "1.8.3")
    bench_name = "benchmark-{ver}".format(ver = bench_ver)
    http_archive(
        name = "com_google_benchmark",
        strip_prefix = bench_name,
        urls = [
            "https://github.com/google/benchmark/archive/v{ver}.tar.gz".format(ver = bench_ver),
        ],
    )
