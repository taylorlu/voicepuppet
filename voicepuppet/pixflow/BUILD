package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "pixflow",
    srcs = ["pixflow.py"],
    deps = [
      "//config:configure",
      "//voicepuppet:builder"
    ],
)

py_binary(
    name = "train_pixflow",
    srcs = ["train_pixflow.py"],
    deps = [
      "//utils:utils",
      ":pixflow",
      "//generator:generator",
      "//generator:loader"
    ],
)

py_binary(
    name = "infer_pixflow",
    srcs = ["infer_pixflow.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:bfm_visual",
      "//utils:utils",
      ":pixflow",
      "//voicepuppet/bfmnet:bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)

py_binary(
    name = "infer_bfm_pixflow",
    srcs = ["infer_bfm_pixflow.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:bfm_visual",
      "//utils:utils",
      ":pixflow",
      "//voicepuppet/bfmnet:bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)
