package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
    ],
)

py_binary(
    name = "makelist_bfm",
    srcs = ["makelist_bfm.py"],
    deps = [
    "//config:configure"
    ],
)

py_binary(
    name = "makelist_pixrefer",
    srcs = ["makelist_pixrefer.py"],
    deps = [
    "//config:configure"
    ],
)

py_binary(
    name = "make_data_from_GRID",
    srcs = ["make_data_from_GRID.py"],
    deps = [
      "//utils:bfm_load_data",
      "//utils:reconstruct_mesh",
      "//utils:utils",
      ":models"
    ],
)