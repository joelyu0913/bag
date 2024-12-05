import importlib


def import_attr(full_name) -> None:
    mod, attr = full_name.rsplit(".", 1)
    mod = importlib.import_module(mod)
    return getattr(mod, attr)
