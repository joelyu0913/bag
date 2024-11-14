from typing import Optional, Union

import numpy as np

from yang.data import Array
from yang.sim.env import Env
from yang.sim.ext import ExprRunner as CppExprRunner


class ExprRunner(object):
    def __init__(
        self,
        env: Env,
        base_data: Optional[list[str]] = None,
        groups: Optional[list[str]] = None,
        extra_data: Optional[list[tuple[str, str, str]]] = None,
        empty_base: bool = False,
    ):
        self.env = env
        if empty_base:
            self.cpp_runner = CppExprRunner(env.cpp_env, [], [], [])
        else:
            self.cpp_runner = CppExprRunner(env.cpp_env)
        if base_data is not None:
            self.cpp_runner.add_base_data(base_data)
        if groups is not None:
            self.cpp_runner.add_groups(groups)
        if extra_data is not None:
            self.cpp_runner.add_extra_data(extra_data)

    def add_base_data(self, base_data: list[str]) -> None:
        self.cpp_runner.add_base_data(base_data)

    def add_groups(self, groups: list[str]) -> None:
        self.cpp_runner.add_groups(groups)

    def add_extra_data(self, extra_data: list[tuple[str, str, str]]) -> None:
        self.cpp_runner.add_extra_data(extra_data)

    def run(
        self,
        expr: str,
        univ: Optional[str] = None,
        output: Optional[Union[np.array, Array]] = None,
        mode: str = "mixed",
        dates_size: int = -1,
        start_di: int = -1,
        end_di: int = -1,
    ) -> Union[np.array, Array]:
        if dates_size < 0:
            dates_size = self.env.dates_size
        if output is None:
            output = np.empty((dates_size, self.env.max_univ_size), dtype=np.float32)
            output_mat = output
        elif isinstance(output, Array):
            output_mat = output.data
        else:
            output_mat = output
        self.cpp_runner.run(expr, univ or "", output_mat, mode, dates_size, start_di, end_di)
        return output

    @staticmethod
    def default_groups():
        return CppExprRunner.default_groups()
