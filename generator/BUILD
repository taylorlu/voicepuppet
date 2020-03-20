package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
    deps = [
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        ":loader",
        "//config:configure"
    ],
)

py_library(
    name = "test_generator",
    srcs = ["test_generator.py"],
    deps = [
        ":generator"
    ],
)
