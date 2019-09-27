#include <RcppEigen.h>
#include "log-log-ex.h"

using Rcpp::List;
using Rcpp::Named;

//' @export
//[[Rcpp::export]]
Eigen::MatrixXd loglog
  (SEXP tobs, SEXP event, SEXP X, SEXP XD,
   double const eps, double const kappa, SEXP beta){
  return loglog_grad(tobs, event, X, XD, eps, kappa, beta);
}

//' @export
//[[Rcpp::export]]
List loglog_opt
  (SEXP tobs, SEXP event, SEXP X, SEXP XD,
   double const eps, double const kappa, SEXP beta){
  auto out = loglog_optim(tobs, event, X, XD, eps, kappa, beta);

  return List::create(
    Named("par")   = out.par,
    Named("value") = out.value);
}
