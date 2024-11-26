#include <pybind11/pybind11.h>

#include "yang/util/config.h"
#include "yang/util/logging.h"

namespace py = pybind11;

PYBIND11_MODULE(ext, m) {
  using yang::Config;

  py::class_<Config>(m, "Config")
      .def(py::init())
      .def("to_yaml_string", &Config::ToYamlString)
      .def_static("load", &Config::Load)
      .def_static("load_file", static_cast<Config (*)(const std::string &)>(&Config::LoadFile));

  m.def("configure_cpp_logging", &yang::ConfigureLogging, py::arg("pattern"),
        py::arg("level") = "info", py::arg("log_stderr") = false, py::arg("log_file") = "");
}
