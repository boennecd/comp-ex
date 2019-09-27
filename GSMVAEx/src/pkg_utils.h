#ifndef PKG_UTILS_H
#define PKG_UTILS_H
#include <cppad/local/declare_ad.hpp>
#include "nloptrAPI.h"
#include <memory>
#include <string>
#include <limits>

namespace pkg_utils {
using std::size_t;

/* class for RAII */
class get_nlopt_problem {
public:
  nlopt_opt opt;
  get_nlopt_problem(const unsigned n){
    /* TODO: allow the user to change these. See
     *   https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
     * for options. */
    opt = nlopt_create(NLOPT_LD_LBFGS, n);

    /* change default of using 10 MB storage for gradient. See
     *  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#Vector_storage_for_limited-memory_quasi-Newton_algorithms.md
     * TODO: allow the user to change this */
    constexpr int n_store(250L);
    nlopt_set_vector_storage(opt, n_store);
  }

  ~get_nlopt_problem(){
    nlopt_destroy(opt);
  }
};

/* util function used by optim */
double optim_ad_fun_double(unsigned int, double const*, double*, void*);

/* optimizes parameters */
struct optim_return {
  double value;
  std::unique_ptr<double[]> par;
};

/* TODO: consider doing something smarter in the one dimensional case */
template<template <class> class vectortype, class T>
optim_return
optim(ADFun<double> &func, vectortype<T> const &start_val){
  size_t const n_indep = func.Domain();
  if(func.Range() != 1)
    throw std::invalid_argument("More than one output in optimizer");
  if((size_t)start_val.size() != n_indep)
    throw std::invalid_argument("Invalid size of starting value");

  /* copy values */
  std::unique_ptr<double[]> vals(new double[n_indep]);
  {
    double *o = vals.get();
    for(unsigned i = 0; i < n_indep; ++i)
      *o++ = asDouble(start_val[i]);
  }

  /* Setup problem. TODO: allow the user to change these... */
  get_nlopt_problem problem(n_indep);
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  nlopt_set_ftol_rel(problem.opt, eps);
  nlopt_set_maxeval(problem.opt, 1000L);
  nlopt_set_min_objective(problem.opt, optim_ad_fun_double, &func);

  /* minimize */
  double min_val;
  {
    auto return_code =
      nlopt_optimize(problem.opt, vals.get(), &min_val);
    if(return_code < 1 or return_code > 4)
      throw std::runtime_error(
        "nlopt return code was " + std::to_string(return_code));
  }

  return { min_val, std::move(vals) };
}

double optim_ad_fun_double
  (unsigned int n, double const *x, double *grad, void *data_in){
  ADFun<double> *func = (ADFun<double>*)data_in;

  /* TODO: can we avoid the copies in this function */
  vector<double> point(n);
  for(unsigned i = 0; i < n; ++i)
    point[i] = *(x + i);

  /* TODO: I think that we may avoid this call if we use forward or
   * something else in CppAD if we have already computed the gradient */
  double const func_val = double(func->Forward(0, point)[0]);

  if(grad){
    auto jac = func->Jacobian(point);
    for(unsigned i = 0; i < n; ++i)
      *(grad + i) = jac[i];

  }

  return func_val;
}
}

#endif
