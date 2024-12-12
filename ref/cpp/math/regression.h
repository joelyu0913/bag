#pragma once

#include <armadillo>

namespace yang::math {

arma::mat LinearRegression(const arma::mat &X, const arma::mat &Y, float lambda = 0);

}
