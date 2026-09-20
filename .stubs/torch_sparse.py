raise ImportError(
    "torch_sparse is deliberately unavailable. The installed build is linked "
    "against a different torch and fails with `undefined symbol: "
    "_ZN5torch3jit17parseSchemaOrNameERKSsb`. PyG catches that ImportError and "
    "falls back to pure-torch scatter, but only AFTER dlopen has already mapped "
    "the broken object into the process. This stub makes the import fail before "
    "any native code is loaded. Nothing here needs the extension."
)
