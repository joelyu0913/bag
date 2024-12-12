py_library(
    name = "prophet",
    srcs = glob(["*.py"]),
    deps = [
        "//python/src/yang/data",
        "//python/src/nu",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "dist",
    srcs = glob(["*.py"], exclude=["nn.py"]),
    visibility = ["//visibility:public"],
)
