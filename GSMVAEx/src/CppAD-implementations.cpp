#include "TMB-wrap.h"

/* we have to declare this somewhere */
template<class Type>
Type objective_function<Type>::operator() ()
{
  return 0;
}

/* log-log example */
#include "log-log-ex.h"

using AD_dub = AD<double>;

#define DAT_MAT(NAME)                                           \
NAME(asMatrix<AD_dub>(NAME))
#define DAT_VEC(NAME)                                           \
NAME(asVector<AD_dub>(NAME))

namespace {
/* would likely have been easier/shorter without a class... */
class loglog_inner_worker {
  /* TODO: Do we need to store these value while we use the ADFun<double>
   * object? It does not appear so from e.g.
   *   https://github.com/kaskr/adcomp/blob/f651356123d20c13063ca64f91f92948c8ef9126/TMB/inst/include/tmb_core.hpp#L1116-L1140
   *
   * In any case, it may be useful if we later want to allow to change some
   * of these and create a new ADFun<double> object. */
  vector<AD_dub> tobs, event;
  matrix<AD_dub> X, XD;
  AD_dub eps, kappa;

public:
  loglog_inner_worker(
    SEXP tobs, SEXP event, SEXP X, SEXP XD, double const eps,
    double const kappa): DAT_VEC(tobs), DAT_VEC(event), DAT_MAT(X),
    DAT_MAT(XD), eps(eps), kappa(kappa) {
    /* TODO: check args */
  }

  ADFun<double> get_func(vector<AD_dub> &beta) const {
    /* Do as in
     *   https://github.com/kaskr/adcomp/blob/f651356123d20c13063ca64f91f92948c8ef9126/TMB/inst/include/tmb_core.hpp#L1116-L1140
     *
     * TODO: Maybe we should copy beta and use the copy instead? Then we can
     * also mark the input as const.
     */
    Independent(beta);

    vector<AD_dub> const eta = X * beta,
                        etaD = XD * beta,
                           h = etaD * exp(eta),
                           H = exp(eta);
    vector<AD_dub> nlogl(1);
    nlogl[0] = 0;
    AD_dub const log_eps = log(eps);
    /* TODO: I guess the `vector` may be like a C++ 'vector' in which case
     * `operator[]` has no bound checks
     *
     * It is an Eigen matrix. I have not used that library much so I need to
     * check */
    for(unsigned i = 0; i < tobs.size(); ++i) {
      nlogl[0] -= event[i] * log_eps - H[i];
      if(h[i] < eps)
        nlogl[0] += h[i] * h[i] * kappa;
    }

    ADFun<double> func(beta, nlogl);
    func.optimize();

    return func;
  }
};
}

Eigen::Array<double, Eigen::Dynamic, 1> loglog_grad
  (SEXP tobs, SEXP event, SEXP X, SEXP XD,
   double const eps, double const kappa, SEXP beta_in){
  loglog_inner_worker worker(tobs, event, X, XD, eps, kappa);

  vector<AD_dub> beta(asVector<AD_dub>(beta_in));
  ADFun<double> func = worker.get_func(beta);

  /* TODO: I think there is a copy here and that can this copy be avoided? */
  vector<double> beta_plain(asVector<double>(beta_in));
  return func.Jacobian(beta_plain);
}
