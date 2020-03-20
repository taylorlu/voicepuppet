package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "pix2pix",
    srcs = ["pix2pix.py"],
    deps = [
      "//config:configure",
      "//voicepuppet:builder"
    ],
)

py_binary(
    name = "train_pix2pix",
    srcs = ["train_pix2pix.py"],
    deps = [
      "//utils:utils",
      ":pix2pix",
      "//generator:generator",
      "//generator:loader"
    ],
)

py_binary(
    name = "infer_pix2pix",
    srcs = ["infer_pix2pix.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:bfm_visual",
      "//utils:utils",
      "//voicepuppet/bfmnet:bfmnet",
      "//generator:generator",
      "//generator:loader"
    ],
)
