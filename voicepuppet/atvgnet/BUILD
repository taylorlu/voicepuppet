package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "backbone",
    srcs = ["backbone.py"],
    deps = [
    ],
)

py_library(
    name = "tinynet",
    srcs = ["tinynet.py"],
    deps = [
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    deps = [
    ],
)

py_library(
    name = "atnet",
    srcs = ["atnet.py"],
    deps = [
        "//config:configure",
        ":tinynet",
        "//voicepuppet:builder"
    ],
)

py_library(
    name = "vgnet",
    srcs = ["vgnet.py"],
    deps = [
        "//config:configure",
        "//voicepuppet:builder"
    ],
)

py_binary(
    name = "train_atnet",
    srcs = ["train_atnet.py"],
    deps = [
      ":atnet",
      ":plot",
      "//generator:generator"
    ],
)

py_binary(
    name = "train_vgnet",
    srcs = ["train_vgnet.py"],
    deps = [
      ":vgnet",
      ":plot",
      "//generator:generator"
    ],
)

py_binary(
    name = "test_atnet",
    srcs = ["test_atnet.py"],
    deps = [
      ":atnet",
      ":tinynet"
    ],
)

py_binary(
    name = "test_vgnet",
    srcs = ["test_vgnet.py"],
    deps = [
      ":vgnet",
      ":atnet",
    ],
)

py_binary(
    name = "infer",
    srcs = ["infer.py"],
    deps = [
      ":vgnet",
      ":tinynet",
      "//config:configure",
      "//generator:loader",
      ":plot"
    ],
)