load("//bazel:cc_utils.bzl", "cc_header_only_library")
load("//bazel:yang.bzl", "yang_cc_library")

# External deps visible to end users. All deps used in public headers
# should be included here.
cc_library(
    name = "external_deps",
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@date//:date",
        "@eigen//:eigen",
        "@yaml-cpp//:yaml-cpp",
        "@fmt//:fmt",
        "@spdlog//:spdlog",
    ] + select({
        "@platforms//os:macos": ["@ranges-v3//:ranges-v3"],
        "//conditions:default": [],
    }),
)

yang_cc_library(
    name = "base",
    hdrs = glob([
        "base/*.h",
    ]),
    srcs = glob([
        "base/*.cpp",
    ]),
    deps = [
        ":external_deps",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "util",
    hdrs = glob([
        "util/*.h",
    ]),
    srcs = glob([
        "util/*.cpp",
    ]),
    deps = [
        ":base",
    ] + select({
        "@platforms//os:macos": [],
        "//conditions:default": ["@conda//:libunwind"],
    }),
    linkopts = [
        "-ldl",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "io",
    hdrs = glob([
        "io/*.h",
    ]),
    srcs = glob([
        "io/*.cpp",
    ]),
    deps = [
        ":base",
        ":util",
        "@conda//:zlib",
        "@conda//:zstd",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "data",
    hdrs = glob([
        "data/*.h",
    ]),
    srcs = glob([
        "data/*.cpp",
    ]),
    deps = [
        ":base",
        ":math",
        ":util",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "math",
    hdrs = glob([
        "math/**/*.h",
    ]),
    srcs = glob([
        "math/**/*.cpp",
    ]),
    deps = [
        ":base",
        ":util",
        "@conda//:armadillo",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "expr",
    hdrs = glob([
        "expr/*.h",
    ]),
    srcs = glob([
        "expr/*.cpp",
    ]),
    deps = [
        ":base",
        ":math",
        ":util",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "sim",
    hdrs = glob([
        "sim/**/*.h",
    ]),
    srcs = glob([
        "sim/**/*.cpp",
    ]),
    deps = [
        ":base",
        ":data",
        ":expr",
        ":io",
        ":math",
        ":util",
        "@thread-pool//:thread-pool",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "cn",
    hdrs = glob([
        "cn/*.h",
    ]),
    srcs = glob([
        "cn/*.cpp",
    ]),
    deps = [
        ":base",
        ":sim",
    ],
    alwayslink = 1,
)

yang_cc_library(
    name = "yang_hdrs",
    hdrs = glob([
        "base/**/*.h",
        "cn/**/*.h",
        "data/**/*.h",
        "expr/**/*.h",
        "io/**/*.h",
        "math/**/*.h",
        "sim/**/*.h",
        "util/**/*.h",
    ]),
)

cc_header_only_library(
    name = "yang_public_hdrs",
    deps = [":yang_hdrs"],
)

cc_library(
    name = "libyang",
    deps = [
        ":base",
        ":cn",
        ":data",
        ":io",
        ":sim",
        ":util",
    ],
)

cc_binary(
    name = "libyang.so",
    deps = [":libyang"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

yang_cc_library(
    name = "yang_deps",
    srcs = [":libyang.so"],
    deps = [
        ":yang_public_hdrs",
        ":external_deps",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "runner",
    srcs = ["runner_main.cpp"],
    deps = [
        ":yang_deps",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@conda//:libjemalloc",
    ],
    linkopts = [
        "-ldl",
        "-ljemalloc",
        "-Wl,-rpath,$$ORIGIN",
    ] + select({
        "@platforms//os:macos": [],
        "//conditions:default": ["-lrt"],
    }),
    visibility = ["//visibility:public"],
)
