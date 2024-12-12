import os, sys
import yaml
import time
import logging

# TODO: the rules are very simple now and do not work in all cases, e.g. live
class ModMeta: 
    def __init__(self):
        self.timestamp = 0
        self.start_date = 0
        self.end_date = 0

class RerunManager:

    def __init__(self,  workdir): 
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.mods = {}
        self.start_date = 0
        self.end_date = 0

    def get_meta_path(self, mod):
        # return (f'{self.workdir}/{mod}').native() + ".yml"
        return f'{self.workdir}/{mod}.yml'


    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def can_skip_run(self, mod, deps): 
        meta = self.get_meta(mod)
        # rerun if different start_date or extend
        if meta.start_date != self.start_date or meta.end_date < self.end_date: 
            return False

        for dep in deps:
            # rerun if any dep changes
            dep_meta = self.get_meta(dep)
            if dep_meta.timestamp > meta.timestamp:
                 return False 
        return True

    def record_before_run(self, mod):
        meta_path = self.get_meta_path(mod)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    def record_run(self, mod):
        self.save_meta(mod)

    def save_meta(self, mod):
        meta = ModMeta()
        meta.timestamp = int(time.time() * 1000)
        meta.start_date = self.start_date
        meta.end_date = self.end_date

        yam= {}
        yam["timestamp"] = meta.timestamp
        yam["start_date"] = meta.start_date
        yam["end_date"] = meta.end_date
        yaml.safe_dump(yam, open(self.get_meta_path(mod), 'w'))
        # std::ofstream ofs(GetMetaPath(mod))
        # ofs << yam.ToYamlString()

        # std::lock_guard guard(mutex_)
        self.mods[mod] = meta

    def get_meta(self, mod):

        # std::lock_guard guard(mutex_)
        if mod in self.mods:
            return self.mods[mod]
        else:
            meta = ModMeta()
            meta_path = self.get_meta_path(mod)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as file:
                        yaml_data = yaml.safe_load(file)

                    meta.timestamp = yaml_data["timestamp"]
                    meta.start_date = yaml_data["start_date"]
                    meta.end_date = yaml_data["end_date"]

                except Exception as ex:
                    logger.error(f"Failed to load meta: {meta_path}")
                    # If an error occurs, re-initialize meta to an empty ModMeta
            # std::lock_guard guard(mutex_)
            self.mods[mod] = meta
            return self.mods[mod]

