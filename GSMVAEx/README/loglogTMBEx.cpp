#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(tobs);
  DATA_VECTOR(event); // double
  DATA_MATRIX(X);
  DATA_MATRIX(XD);
  DATA_SCALAR(eps);   // boundary value for values that are too small or negative
  DATA_SCALAR(kappa); // scale for the quadratic penalty
  PARAMETER_VECTOR(beta);
  vector<Type> eta = X*beta;
  vector<Type> etaD = XD*beta;
  vector<Type> h = etaD*exp(eta);
  vector<Type> H = exp(eta);
  // vector<Type> logl = event*log(h)-H;
  vector<Type> logl(tobs.size());
  Type pen = 0.0;
  for(int i=0; i<tobs.size(); ++i) {
    if (h(i)<eps) {
      logl(i) = event(i)*log(eps)-H(i);
      pen += h(i)*h(i)*kappa;
    } else {
      logl(i) = event(i)*log(h(i))-H(i);
    }
  }
  Type f = -sum(logl) + pen;
  ADREPORT(logl); // gradients are the scores (slows sdreport())
  return f;
}
