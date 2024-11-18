import os, sys
import yaml

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

    def GetMetaPath(mod):
        return (workdir_ / mod).native() + ".yml"


    def SetDates(start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def CanSkipRun(mod, deps): 
        meta = self.GetMeta(mod)
        # rerun if different start_date or extend
        if meta.start_date != start_date_ or meta.end_date < self.end_date: 
            return false

        for dep in deps:
            # rerun if any dep changes
            dep_meta = self.GetMeta(dep)
            if dep_meta.timestamp > meta.timestamp:
                 return False 
        return True

    def RecordBeforeRun(mod):
        meta_path = self.GetMetaPath(mod)
        if os.path.exists(meta_path):
            os.remove(file_path)

    def RecordRun(mod):
        SaveMeta(mod)

    def SaveMeta(mod):
        meta = ModMeta()
        meta.timestamp = GetTimestamp()
        meta.start_date = start_date_
        meta.end_date = end_date_

        yam= {}
        yam.Set("timestamp", meta.timestamp)
        yam.Set("start_date", meta.start_date)
        yam.Set("end_date", meta.end_date)
        yaml.safe_dump(yam, self.GetMetaPath(mod))
        # std::ofstream ofs(GetMetaPath(mod))
        # ofs << yam.ToYamlString()

        # std::lock_guard guard(mutex_)
        self.mods[mod] = meta

    def GetMeta(mod):

        # std::lock_guard guard(mutex_)
        it = self.mods.find(mod)
        if mod in self.mods:
            return self.mods[mod]

        meta = ModMeta()
        meta_path = self.GetMetaPath(mod)
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
                meta = ModMeta()
        # std::lock_guard guard(mutex_)
        if mod in self.mods:
            return False
        else:
            self.mods[mod] = meta
            return True

