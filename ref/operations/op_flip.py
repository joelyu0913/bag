from yang.sim.operation import Operation


# in ycfg:
# register_operation("flip", "yao.B.basic.operations.op_flip.OpFlip")
# py_combo(
#     name="op_test",
#     children=["child1"],
#     ops=[
#         {"name": "flip", "args": ["1"], "kwargs": {"a":"1"}}
#     ],
# )
#
# in notebook:
# from yang.sim.operation import register_operation
# import yao.B.basic.operations.op_flip
# register_operation("flip", yao.B.basic.operations.op_flip.OpFlip)
class OpFlip(Operation):
    def apply(self, sig, env, start_di, end_di, args, kwargs):
        sig[start_di:end_di, : env.univ_size] = -sig[start_di:end_di, : env.univ_size]
