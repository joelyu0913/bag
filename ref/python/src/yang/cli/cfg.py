#!/usr/bin/env python3

import fnmatch
import os
import re
from collections.abc import Callable
from string import Template
from typing import Any, Union

import click
import yaml


def substitute_cfg(mappings: dict, cfg: dict):
    if isinstance(cfg, list):
        return [substitute_cfg(mappings, x) for x in cfg]
    elif isinstance(cfg, dict):
        return {k: substitute_cfg(mappings, v) for k, v in cfg.items()}
    elif isinstance(cfg, str):
        tpl = Template(cfg)
        return tpl.safe_substitute(mappings)
    else:
        return cfg


class CfgContext(object):
    modules: list[dict]
    module_map: dict[str, int]
    skip_names: set[str]
    skip_tags: set[str]
    skip_patterns: list[re.Pattern]
    libs: set[str]
    raw_cfg: dict[str, Any]
    aliases: dict[str, str]
    exports: dict[str, Any]
    exec_globals: dict[str, Any]
    post_processors: list[Callable[[], None]]

    def __init__(self, prod_mode: bool = False):
        self.prod_mode = prod_mode
        self.modules = []
        self.module_map = {}
        self.skip_names = set()
        self.skip_tags = set()
        self.skip_patterns = []
        self.libs = set()
        self.py_operations = {}
        self.raw_cfg = {}
        self.aliases = {}
        self.post_processors = []
        self.loaded_files = set()
        self.exports = {
            "use_yaml": self.use_yaml,
            "use": self.use,
            "require": self.require,
            "mod": self.mod,
            "lib": self.lib,
            "opt": self.opt,
            "opt2": self.opt2,
            "alias": self.alias,
            "export": self.export,
            "env": self.env,
            "skip": self.skip,
            "resolve_relative_path": self.resolve_relative_path,
            "register_operation": self.register_operation,
        }
        self.exec_globals = {}

    def resolve_relative_path(self, path: str) -> str:
        if "__file__" in self.exec_globals:
            return os.path.join(os.path.dirname(self.exec_globals["__file__"]), path)
        return path

    def load_yaml(self, yaml_path: str, relative: bool = False) -> str:
        if relative:
            yaml_path = self.resolve_relative_path(yaml_path)
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def use_yaml(self, yaml_path: str, relative: bool = False) -> None:
        cfg = self.load_yaml(yaml_path, relative)
        if "aliases" in cfg:
            self.aliases.update(cfg.pop("aliases"))
        self.raw_cfg.update(cfg)

    def mod(
        self,
        name: str,
        cls: str,
        lang: str,
        stages: list[str] = ["intraday"],
        yaml_cfg: str = None,
        yaml_relative: bool = False,
        post: bool = False,
        tags: list[str] = [],
        **kwargs,
    ) -> None:
        m = {
            "name": name,
            "class": cls,
            "lang": lang,
            "stages": stages,
            "post": post,
            "tags": tags,
            **kwargs,
        }
        if yaml_cfg is not None:
            m.update(self.load_yaml(yaml_cfg, yaml_relative))
        if name in self.module_map:
            idx = self.module_map[name]
            self.modules[idx] = m
        else:
            self.modules.append(m)
            self.module_map[name] = len(self.modules) - 1

    def lib(self, path: Union[list[str], str]) -> None:
        if isinstance(path, list):
            for p in path:
                self.libs.add(p)
        else:
            self.libs.add(path)

    def register_operation(self, name: str, op_cls: str) -> None:
        self.py_operations[name] = op_cls

    def opt(self, **kwargs) -> None:
        self.raw_cfg.update(kwargs)

    def opt2(self, name: Union[list[str], str], **kwargs) -> None:
        if isinstance(name, str):
            path = name.split(".")
        else:
            path = name
        cfg = self.raw_cfg
        for p in path:
            if p in cfg:
                cfg = cfg[p]
            else:
                cfg[p] = {}
                cfg = cfg[p]
        cfg.update(kwargs)

    def alias(self, name: str, value: str) -> None:
        self.aliases[name] = value

    def export(self, name: str, value: Any) -> None:
        self.exports[name] = value

    def post_process(self, f: Callable[[], None]) -> None:
        self.post_processors.append(f)

    def use(
        self, cfg_path: str, relative: bool = False, local_globals: dict = None, **kwargs
    ) -> None:
        if relative:
            cfg_path = self.resolve_relative_path(cfg_path)
        self.loaded_files.add(cfg_path)
        with open(cfg_path) as f:
            cfg_py = f.read()
        code = compile(cfg_py, cfg_path, "exec")

        # create globals dict for sub config
        orig_exec_globals = self.exec_globals
        self.exec_globals = {**self.exports, **kwargs}
        self.exec_globals["__file__"] = cfg_path
        exec(code, self.exec_globals)
        # reset globals dict and add new exports
        self.exec_globals = orig_exec_globals
        self.exec_globals.update(self.exports)
        if local_globals is not None:
            local_globals.update(self.exports)

    def require(self, cfg_path: str, relative: bool = False, local_globals: dict = None) -> bool:
        if relative:
            cfg_path = self.resolve_relative_path(cfg_path)
        if cfg_path in self.loaded_files:
            return False
        self.use(cfg_path, local_globals=local_globals)

    def env(self, key: str, fallback: Any = "") -> str:
        if key == "prod":
            return self.prod_mode
        return os.environ.get(key, fallback)

    def skip(self, name: str = None, tag: str = None) -> None:
        if name:
            if "*" in name:
                self.skip_patterns.append(re.compile(fnmatch.translate(name)))
            else:
                self.skip_names.add(name)
        if tag:
            self.skip_tags.add(tag)

    def to_cfg(self) -> dict:
        for f in self.post_processors:
            f()
        cfg = {**self.raw_cfg}
        cfg["modules"] = self.modules
        cfg["libs"] = list(self.libs)
        cfg["py_operations"] = self.py_operations
        skip_mods = []
        for mod in self.modules:
            if (
                mod["name"] in self.skip_names
                or any(t in self.skip_tags for t in mod.get("tags", []))
                or any(pat.fullmatch(mod["name"]) for pat in self.skip_patterns)
            ):
                skip_mods.append(mod["name"])
        cfg["skip_modules"] = skip_mods
        return substitute_cfg(self.aliases, cfg)


@click.command()
@click.argument("cfg_path")
def main(cfg_path: str) -> None:
    ctx = CfgContext()
    ctx.use(cfg_path)
    print(yaml.dump(ctx.to_cfg()))


if __name__ == "__main__":
    main()
