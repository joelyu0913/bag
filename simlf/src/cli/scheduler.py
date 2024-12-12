import logging
import os

# control run process based on sys/user_cache/_rerun/
class Scheduler(object):
    def __init__(self, config: dict, mods: list[str]):
        self.config = config
        self._run_mods = set(mods)
        self._deps = {}
        self._mods_by_lang = set()
        self._mods_config = {}
        self._preprocess()
        self.num_pending = len(mods)

    # make sure deps exist
    def _preprocess(self) -> None:
        sys_cache = self.config.get("sys_cache")
        user_cache = self.config.get("user_cache")
        sys_mods = set()
        if sys_cache and user_cache and sys_cache != user_cache:
            sys_rerun_dir = os.path.join(sys_cache, "_rerun")
            if os.path.exists(sys_rerun_dir):
                sys_mods = set(
                    os.path.basename(p)[:-4]
                    for p in os.listdir(sys_rerun_dir)
                    if p.endswith(".yml")
                )
            conflict_mods = [m["name"] for m in self.config["modules"] if m["name"] in sys_mods]
            if conflict_mods:
                raise RuntimeError(
                    f"some user modules already exist in sys modules: {conflict_mods}"
                )

        for mod_config in self.config["modules"]:
            name = mod_config["name"]
            if name not in self._run_mods:
                continue
            self._mods_config[name] = mod_config
            self._mods_by_lang.add(name)
            self._deps[name] = mod_config.get("deps", [])

        dep_missing = False
        for mod_name, mod_deps in self._deps.items():
            config = self._mods_config[mod_name]
            if not config.get("validate_deps", True):
                continue
            for dep in mod_deps:
                if dep not in self._deps and dep not in sys_mods:
                    logging.error("%s dep module not found: %s", mod_name, dep)
                    dep_missing = True
        if dep_missing:
            raise RuntimeError("deps not found")

    def pop_ready_modules(self) -> list[str]:

        collected = list(self._mods_by_lang)

        for mod in collected:
            del self._mods_config[mod]
            self._mods_by_lang.remove(mod)
            self.num_pending -= 1
        return collected

    @property
    def finished(self) -> bool:
        return self.num_pending == 0
