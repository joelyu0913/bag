import numpy as np


def in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return True
        else:
            return False
    except NameError:
        return False


def import_attr(full_name):
    import importlib

    mod, attr = full_name.rsplit(".", 1)
    mod = importlib.import_module(mod)
    return getattr(mod, attr)
