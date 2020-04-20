package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "vgg_simple",
    srcs = ["vgg_simple.py"],
    deps = [
      "//config:configure",
      "//voicepuppet:builder"
    ],
)

py_library(
    name = "pixrefer",
    srcs = ["pixrefer.py"],
    deps = [
      "//config:configure",
      "//voicepuppet:builder",
      ":vgg_simple"
    ],
)

py_binary(
    name = "train_pixrefer",
    srcs = ["train_pixrefer.py"],
    deps = [
      "//utils:utils",
      ":pixrefer",
      "//generator:generator",
      "//generator:loader"
    ],
)

py_binary(
    name = "infer_pixrefer",
    srcs = ["infer_pixrefer.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:bfm_visual",
      "//utils:utils",
      ":pixrefer",
      "//voicepuppet/bfmnet:bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)

py_binary(
    name = "infer_bfmvid",
    srcs = ["infer_bfmvid.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:bfm_visual",
      "//utils:utils",
      ":pixrefer",
      "//voicepuppet/bfmnet:bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)
