load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "ext",
    srcs = [
        "ext.cpp",
    ],
    deps = [
        "//cpp/src/yang:yang_deps",
        "//python/src/yang/math:convert",
    ],
)

py_library(
    name = "sim",
    srcs = glob(["*.py"]),
    deps = [
        "//python/src/yang/data",
        "//python/src/yang/util",
    ],
    data = [
        ":ext.so",
    ],
    visibility = ["//visibility:public"],
)
