load("//bazel:cython_library.bzl", "pyx_library")

cc_library(
    name = "ops_impl",
    hdrs = [
        "np_ufuncs.h",
    ],
    srcs = [
        "helpers.h",
        "np_ufuncs_1d.cpp",
        "np_ufuncs_2d.cpp",
        "np_ufuncs_3d.cpp",
        "np_ufuncs_reduce.cpp",
    ],
    deps = [
        "//cpp/src/yang:yang_deps",
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert",
    hdrs = [
        "convert.h",
    ],
    deps = [
        "//cpp/src/yang:yang_deps",
    ],
    visibility = ["//visibility:public"],
)

pyx_library(
    name = "ops",
    srcs = [
        "ops.pyx",
    ],
    deps = [
        ":ops_impl",
    ],
)

py_library(
    name = "math",
    srcs = glob(["*.py"]),
    deps = [
        ":ops",
    ],
    visibility = ["//visibility:public"],
)
