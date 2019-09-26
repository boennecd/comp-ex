#include <RcppEigen.h>
#include "log-log-ex.h"

//' @export
//[[Rcpp::export]]
void hello_world_func(){
  Rcpp::Rcout << "Hello world\n";
}

//' @export
//[[Rcpp::export]]
Eigen::MatrixXd loglog
  (SEXP tobs, SEXP event, SEXP X, SEXP XD,
   double const eps, double const kappa, SEXP beta){
  return loglog_grad(tobs, event, X, XD, eps, kappa, beta);
}
