#include "yang/math/eigen.h"
#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpAdd : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float val) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        sig(di, ii) += val;
      }
    }
  }
};

struct OpMultiply : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float val) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        sig(di, ii) *= val;
      }
    }
  }
};

struct OpLog : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di) final {
    auto mat = mat_to_eigen(sig).block(start_di, 0, end_di - start_di, env.univ_size());
    mat = Eigen::log(mat);
  }
};

struct OpNeg : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di) final {
    auto mat = mat_to_eigen(sig).block(start_di, 0, end_di - start_di, env.univ_size());
    mat = -mat;
  }
};

struct OpInverse : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di) final {
    auto mat = mat_to_eigen(sig).block(start_di, 0, end_di - start_di, env.univ_size());
    mat = 1 / mat;
  }
};

REGISTER_OPERATION("neg", OpNeg);
REGISTER_OPERATION("add", OpAdd);
REGISTER_OPERATION("multiply", OpMultiply);
REGISTER_OPERATION("log", OpLog);
REGISTER_OPERATION("inverse", OpInverse);

}  // namespace
