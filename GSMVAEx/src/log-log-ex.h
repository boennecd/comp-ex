#ifndef LOG_LOG_EX
#define LOG_LOG_EX

#include <Eigen/Dense>
#include <Rinternals.h>

/* returns a vector<double> from TMB. See
 *   https://github.com/kaskr/adcomp/blob/f651356123d20c13063ca64f91f92948c8ef9126/TMB/inst/include/tmbutils/vector.hpp#L17
 */
using eigen_mat_vec = Eigen::Array<double, Eigen::Dynamic, 1>;

eigen_mat_vec loglog_grad
  (SEXP, SEXP, SEXP, SEXP, double const, double const, SEXP);

struct loglog_optim_return {
  double value;
  eigen_mat_vec par;
};

loglog_optim_return loglog_optim
  (SEXP, SEXP, SEXP, SEXP, double const, double const, SEXP);

#endif
