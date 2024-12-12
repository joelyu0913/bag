load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "ext",
    srcs = [
        "ext.cpp",
    ],
    deps = [
        "//cpp/src/yang:yang_deps",
    ],
)

py_library(
    name = "util",
    srcs = glob(["*.py"]),
    data = [
        ":ext.so",
    ],
    visibility = ["//visibility:public"],
)
