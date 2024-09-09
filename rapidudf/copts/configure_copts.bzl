RUDF_DEFAULT_COPTS = [
    "-D__STDC_FORMAT_MACROS",
    "-D__STDC_LIMIT_MACROS",
    "-D__STDC_CONSTANT_MACROS",
    "-DGFLAGS_NS=google",
    "-Werror=return-type",
]

RUDF_DEFAULT_LINKOPTS = [
    "-L/usr/local/lib",
    "-L/usr/local/lib64",
    # "-lfmt",
    "-ldl",
    "-lrt",
]
