from collections import deque
from typing import Optional


class Scheduler(object):
    def __init__(self, tasks: list[str], dep_map: dict[str, list[str]]):
        self.tasks = tasks
        self._preprocess(dep_map)
        self.error = []
        self.finished = False

    def _preprocess(self, dep_map: dict[str, list[str]]) -> None:
        self._in_degree = [0 for _ in self.tasks]
        self._out_edges = [[] for _ in self.tasks]
        self._id_map = {}
        for i, task_name in enumerate(self.tasks):
            self._id_map[task_name] = i
        for consumer, deps in dep_map.items():
            if consumer not in self._id_map:
                continue
            consumer_id = self._id_map[consumer]
            for producer in deps:
                if producer not in self._id_map:
                    continue
                producer_id = self._id_map[producer]
                self._in_degree[consumer_id] += 1
                self._out_edges[producer_id].append(consumer_id)

        self._ready_tasks = deque()
        for i in range(len(self.tasks)):
            if self._in_degree[i] == 0:
                self._ready_tasks.append(i)
        self._finished_tasks = set()
        self.finished = len(self.tasks) == 0

    @property
    def has_ready_tasks(self) -> bool:
        return len(self._ready_tasks) > 0

    def next_ready_task(self) -> Optional[str]:
        if self._ready_tasks:
            return self.tasks[self._ready_tasks.popleft()]

    def on_task_finished(self, task_name: str) -> None:
        task_id = self._id_map[task_name]
        if task_id in self._finished_tasks:
            return
        self._finished_tasks.add(task_id)
        for consumer_id in self._out_edges[task_id]:
            self._in_degree[consumer_id] -= 1
            if self._in_degree[consumer_id] == 0:
                self._ready_tasks.append(consumer_id)
        self.finished = len(self._finished_tasks) == len(self.tasks)

    def on_task_error(self, task_name: str) -> None:
        self.error.append(task_name)

    def get_pending_tasks(self) -> list[str]:
        return [self.tasks[i] for i in range(len(self.tasks)) if i not in self._finished_tasks]
