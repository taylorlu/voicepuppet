package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "tinynet",
    srcs = ["tinynet.py"],
    deps = [
    ],
)

py_library(
    name = "bfmnet",
    srcs = ["bfmnet.py"],
    deps = [
        "//config:configure",
        ":tinynet",
        "//voicepuppet:builder"
    ],
)

py_binary(
    name = "train_bfmnet",
    srcs = ["train_bfmnet.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:bfm_visual",
      "//utils:reconstruct_mesh",
      "//utils:utils",
      ":bfmnet",
      "//generator:generator"
    ],
)

py_binary(
    name = "infer_bfmnet",
    srcs = ["infer_bfmnet.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:bfm_visual",
      "//utils:reconstruct_mesh",
      "//utils:utils",
      ":bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)
