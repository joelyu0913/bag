py_library(
    name = "base_path",
    imports = ["."],
    visibility = ["//visibility:public"],
)

# dummy rule to gather all py targets
py_binary(
    name = "all",
    srcs = ["all.py"],
    deps = [
        "//python/src/nu",
        "//python/src/prophet",
        "//python/src/yang/cli",
        "//python/src/yang/data",
        "//python/src/yang/math",
        "//python/src/yang/sim",
        "//python/src/yang/util",
        "//yao:yao_py",
    ],
)
