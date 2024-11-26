#include "yang/math/regression.h"

#include <cmath>
#include <vector>

#include "yang/util/logging.h"

namespace yang::math {

arma::mat LinearRegression(const arma::mat &X, const arma::mat &Y, float lambda) {
  ENSURE2(Y.n_cols == 1 && X.n_rows == Y.n_rows);

  std::vector<int> index;
  index.reserve(X.n_rows);
  for (int rid = 0; rid < static_cast<int>(X.n_rows); rid++) {
    bool valid_row = true;
    for (int cid = 0; cid < static_cast<int>(X.n_cols); cid++) {
      if (!std::isfinite(X(rid, cid))) {
        valid_row = false;
        break;
      }
    }
    if (valid_row && std::isfinite(Y(rid))) index.push_back(rid);
  }
  arma::uvec index_vec(index.size());
  for (int i = 0; i < static_cast<int>(index.size()); ++i) index_vec[i] = index[i];
  arma::mat Xn = X.rows(index_vec);
  arma::mat Yn = Y.rows(index_vec);
  arma::mat I2(Xn.n_cols, Xn.n_cols, arma::fill::eye);
  arma::mat XCor = Xn.t() * Xn + lambda * I2;
  auto det = std::abs(arma::det(XCor));
  if (det < 1e-100 || !std::isfinite(det)) {
    // LOG_WARN("det not valid: {}", det);
    // return arma::mat(X.n_cols, 1, arma::fill::zeros); // previous version
    LOG_DEBUG("det not valid: {}", det);
    return arma::mat(X.n_cols, 1, arma::fill::value(NAN));
  }
  return XCor.i() * Xn.t() * Yn;
}

}  // namespace yang::math
