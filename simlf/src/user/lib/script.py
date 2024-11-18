import os


def get_project_root():
    proj_dir = os.environ.get("YANG_ROOT")
    if proj_dir is not None:
        return proj_dir
    proj_dir = os.getcwd()
    while not os.path.exists(os.path.join(proj_dir, "WORKSPACE")):
        if proj_dir == "/":
            raise RuntimeError("project root not found")
        proj_dir = os.path.dirname(proj_dir)
    return proj_dir


def load_libyao(proj_root: str = None):
    import yang.sim

    if proj_root is None:
        proj_root = get_project_root()
    yang.sim.load_shared_lib(f"{proj_root}/lib/libyao.so")


def get_env(cache_dir: str):
    proj_dir = get_project_root()
    if not cache_dir.startswith("/") and not os.path.exists(cache_dir):
        cache_dir = os.path.join(proj_dir, cache_dir)
    load_libyao(proj_dir)
    return yang.sim.Env({"cache": cache_dir})
