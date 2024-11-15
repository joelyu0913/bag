import numpy as np

from sim.env import Env, cast_cpp_env
from sim.ext import Operation as CppOperation
from sim.ext import OperationFactory as CppOperationFactory
from sim.ext import set_py_operation_factory


class OperationFactory(CppOperationFactory):
    def __init__(self):
        CppOperationFactory.__init__(self)
        self.id_cnt = 0
        self.op_classes = {}
        self.ops = {}

    def make(self, name: str):
        op_cls = self.op_classes.get(name)
        if op_cls:
            op = op_cls()
            self.id_cnt += 1
            op_id = self.id_cnt
            self.ops[op_id] = op
            return (op_id, op)
        else:
            return (0, None)

    def free(self, op_id: int, op):
        self.ops.pop(op_id)

    def register_operation(self, name: str, op_cls):
        self.op_classes[name] = op_cls


OPERATION_FACTORY = OperationFactory()
set_py_operation_factory(OPERATION_FACTORY)


class Operation(CppOperation):
    def apply_raw(self, sig, env, start_di, end_di, args, kwargs):
        sig_array = np.array(sig, copy=False)
        self.apply(sig_array, cast_cpp_env(env), start_di, end_di, args, kwargs)

    def apply(
        self,
        sig: np.array,
        env: Env,
        start_di: int,
        end_di: int,
        args: list[str],
        kwargs: dict[str, str],
    ):
        pass


def register_operation(name: str, op_cls):
    OPERATION_FACTORY.register_operation(name, op_cls)
