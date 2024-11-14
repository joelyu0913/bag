#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "python/src/yang/math/convert.h"
#include "yang/sim/env.h"
#include "yang/sim/expr_runner.h"
#include "yang/sim/operation_manager.h"
#include "yang/util/logging.h"
#include "yang/util/module_loader.h"

namespace py = pybind11;

namespace yang {

class PyOperation : public Operation {
 public:
  bool lookback() const override {
    PYBIND11_OVERRIDE_NAME(bool, Operation, "lookback", lookback, );
  }

  void Apply(MatView<float> sig, const Env &env, int start_di, int end_di,
             const std::vector<std::string> &args,
             const std::unordered_map<std::string, std::string> &kwargs) override {
    PYBIND11_OVERRIDE_NAME(void, Operation, "apply_raw", Apply, sig, env, start_di, end_di, args,
                           kwargs);
  }
};

class PyOperationFactory : public OperationFactory {
 public:
  using OpRet = std::pair<int, Operation *>;

  OpRet Make(std::string_view name) override {
    PYBIND11_OVERRIDE_PURE_NAME(OpRet, OperationFactory, "make", Make, name);
  }

  void Free(int id, Operation *op) override {
    PYBIND11_OVERRIDE_PURE_NAME(void, OperationFactory, "free", Free, id, op);
  }
};

class PyOperationManager {
 public:
  void Initialize(const Env *env) {
    underlying_.Initialize(env);
  }

  void Apply(py::buffer sig_in, py::buffer sig_out, int start_di, int end_di,
             const std::vector<OperationDesc> &ops) {
    auto mat_in = math::py_buffer_to_mat<const float>(sig_in, "f");
    auto mat_out = math::py_buffer_to_mat<float>(sig_out, "f");
    py::gil_scoped_release release;
    underlying_.Apply(mat_in, mat_out, start_di, end_di, ops);
  }

 private:
  OperationManager underlying_;
};

void PyLoadSharedLib(const std::string &path) {
  ModuleLoader::Load(path);
}

class PyExprRunner : public ExprRunner {
 public:
  using ExprRunner::ExprRunner;

  void Run(std::string_view expr_str, std::string_view univ, py::buffer output,
           std::string_view mode, int dates_size = -1, int start_di = -1, int end_di = -1) {
    auto info = output.request();
    auto output_mat = info.format == "f"
                          ? expr::OutFloatMat(math::py_buffer_to_mat<float>(output, "f"))
                          : expr::OutFloatMat(math::py_buffer_to_mat<double>(output, "d"));
    if (dates_size < 0) dates_size = env_->dates_size();
    if (start_di < 0) start_di = env_->start_di();
    if (end_di < 0) end_di = env_->end_di();
    py::gil_scoped_release release;
    return ExprRunner::Run(expr_str, univ, output_mat, ExprRunner::ParseMode(mode), dates_size,
                           start_di, end_di);
  }
};

}  // namespace yang

PYBIND11_MODULE(ext, m) {
  using namespace yang;
  using namespace pybind11::literals;

  m.def("load_shared_lib", &PyLoadSharedLib);
  m.def("set_py_operation_factory", &SetPyOperationFactory);

  py::class_<DataCache>(m, "DataCache");

  py::class_<RerunManager>(m, "RerunManager")
      .def("can_skip_run", &RerunManager::CanSkipRun, py::call_guard<py::gil_scoped_release>())
      .def("record_before_run", &RerunManager::RecordBeforeRun,
           py::call_guard<py::gil_scoped_release>())
      .def("record_run", &RerunManager::RecordRun, py::call_guard<py::gil_scoped_release>());

  py::class_<Env>(m, "Env")
      .def(py::init())
      .def("initialize", &Env::Initialize)
      .def("load", &Env::Load)
      .def_property_readonly("data_cache", &Env::data_cache)
      .def_property_readonly("rerun_manager", &Env::rerun_manager)
      .def("get_config", [](const Env &self) { return self.config(); })
      .def_property_readonly("addr",
                             [](const Env &self) { return reinterpret_cast<uint64_t>(&self); });

  py::class_<math::MatView<float>>(m, "FloatMat", pybind11::buffer_protocol())
      .def_buffer([](math::MatView<float> &m) {
        return py::buffer_info(m.data(), sizeof(float), py::format_descriptor<float>::format(), 2,
                               {m.rows(), m.cols()}, {sizeof(float) * m.cols(), sizeof(float)});
      });

  py::class_<Operation, PyOperation>(m, "Operation")
      .def(py::init<>())
      .def("lookback", &Operation::lookback)
      .def("apply", py::overload_cast<math::MatView<float>, const Env &, int, int,
                                      const std::vector<std::string> &>(&Operation::Apply));

  py::class_<OperationFactory, PyOperationFactory>(m, "OperationFactory")
      .def(py::init<>())
      .def("make", &OperationFactory::Make)
      .def("free", &OperationFactory::Free);

  py::class_<PyOperationManager>(m, "OperationManager")
      .def(py::init())
      .def("initialize", &PyOperationManager::Initialize, "env"_a)
      .def("apply", &PyOperationManager::Apply);

  py::class_<PyExprRunner>(m, "ExprRunner")
      .def(py::init<const Env *>())
      .def(py::init<const Env *, const std::vector<std::string> &, const std::vector<std::string> &,
                    const std::vector<ExprRunner::DataInfo> &>())
      .def("add_base_data", &PyExprRunner::AddBaseData)
      .def("add_groups", &PyExprRunner::AddGroups)
      .def("add_extra_data", &PyExprRunner::AddExtraData)
      .def("run", &PyExprRunner::Run, "expr_str"_a, "univ"_a, "output"_a, "mode"_a = "mixed",
           "dates_size"_a = -1, "start_di"_a = -1, "end_di"_a = -1)
      .def_static("default_base_data", &PyExprRunner::default_base_data)
      .def_static("default_groups", &PyExprRunner::default_groups);
}
