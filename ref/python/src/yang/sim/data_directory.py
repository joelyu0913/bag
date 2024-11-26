import os
from typing import Optional


class DataDirectory(object):
    def __init__(self, user_dir: str, sys_dir: str = None):
        self.user_dir = user_dir
        self.sys_dir = sys_dir or user_dir
        os.makedirs(self.user_dir, exist_ok=True)

    def get_path(self, mod: str, data: Optional[str] = None):
        return self.get_read_path(mod, data)

    def get_read_path(self, mod: str, data: Optional[str] = None):
        if data:
            user_path = os.path.join(self.user_dir, mod, data)
        else:
            user_path = os.path.join(self.user_dir, mod)
        if os.path.exists(user_path):
            return user_path
        if data:
            sys_path = os.path.join(self.sys_dir, mod, data)
        else:
            sys_path = os.path.join(self.sys_dir, mod)
        if (
            os.path.exists(sys_path)
            or os.path.exists(sys_path + ".meta")
            or os.path.exists(sys_path + ".id")
        ):
            return sys_path
        return user_path

    def get_write_path(self, mod: str, data: Optional[str] = None):
        if data:
            return os.path.join(self.user_dir, mod, data)
        else:
            return os.path.join(self.user_dir, mod)
