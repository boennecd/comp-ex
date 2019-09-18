#include <boost/math/differentiation/autodiff.hpp>
#include <RcppArmadillo.h>
#include <stdexcept>
#include <vector>
#include <cmath>

using namespace boost::math::differentiation;
using std::size_t;

namespace {
/* class to evaluate lower bound terms, gradient of lower bound, etc.
 * for each group */
class lb_group {
  size_t const grp_size;
  double const * const y_start, * const x_start;

  /* function to evalute LB */
  template<typename L, typename M, typename S, typename B0, typename B1>
  promote<L, M, S, B0, B1> lb_wo_constant_inner
    (L const &lambda_log, M const &mu, S const &sigma_log,
     B0 const &b0, B1 const &b1) const {
    auto const sigma  = exp(sigma_log); /* TODO: could be moved */
    auto const lambda = exp(lambda_log),
          lambda_half = lambda * .5;

    auto const term1 =
      .5 * (lambda_log - (mu * mu + lambda) / (sigma * sigma))
      /* TODO: could be moved */
      - sigma_log;


    /* compute terms from conditional density */
    auto get_lp_term = [&](double const y, double const x){
      auto const lp = b0 + b1 * x + mu;
      return y * lp - exp(lp + lambda_half);
    };

    /* assume that there is at least one term... */
    double const * yi = y_start, * xi = x_start;
    auto term2 = get_lp_term(*yi++, *xi++);
    for(size_t i = 1; i < grp_size; ++i)
      term2 += get_lp_term(*yi++, *xi++);

    return term1 + term2;
  }

public:
  /* normalization constant. Can make evaluations cheaper if we make
   * repeated calls with different parameters */
  double const norm_constant = ([&](){
    double out(-0.693147180559945) /* log(1/2) */;
    double const *y_i = y_start;
    for(size_t i = 0; i < grp_size; ++i)
      out -= std::lgamma(*y_i++ + 1);
    return out;
  })();

  lb_group
  (size_t const grp_size, double const * const y_start,
   double const * const x_start):
  grp_size(grp_size), y_start(y_start), x_start(x_start) { }

  template<size_t Ol,
           typename ::size_t Om  = Ol, typename ::size_t Os = Ol,
           typename ::size_t Ob0 = Ol, typename ::size_t Ob1 = Ol >
  auto lb_wo_constant
    (double const &lambda_log, double const &mu, double const &sigma_log,
     double const &b0, double const &b1) const {
    auto const vars = make_ftuple<double, Ol, Om, Os, Ob0, Ob1>(
      lambda_log, mu, sigma_log, b0, b1);
    auto const &vla = std::get<0>(vars);
    auto const &vmu = std::get<1>(vars);
    auto const &vsi = std::get<2>(vars);
    auto const &vb0 = std::get<3>(vars);
    auto const &vb1 = std::get<4>(vars);

    return lb_wo_constant_inner(vla, vmu, vsi, vb0, vb1);
  }

  /* could make a method to optimize (mu, lambda) given (sigma, beta) e.g.,
   * using nloptr which has a C interface */
};

/* util class to compute given order derivatives */
template<size_t order>
class comp_out {
public:
  double lb;
  arma::vec gr;

  comp_out
    (std::vector<lb_group> const &grps, arma::vec const &lambda_log,
     arma::vec const &mu, double const sigma_log, arma::vec const &beta) {
    lb = 0;
    if(order > 0)
      gr.zeros(2 * grps.size() + 3);
    size_t im = 3, is = grps.size() + 3, i = 0L;

    for(auto const &g : grps){
      {
        /* TODO: this is a bit stupid if order > 0... */
        auto obj = g.lb_wo_constant<0>(
          lambda_log[i], mu[i], sigma_log, beta[0], beta[1]);
        lb += obj.derivative(0, 0, 0, 0, 0) + g.norm_constant;
      }

      if(order > 0){
        /* lambda to get first order partial derivative */
        auto func = [&](unsigned const arg){
          if(arg == 0)
            return g.lb_wo_constant<1, 0, 0, 0, 0>
              (lambda_log[i], mu[i], sigma_log, beta[0], beta[1]).
               derivative(1, 0, 0, 0, 0);
          else if(arg == 1)
            return g.lb_wo_constant<0, 1, 0, 0, 0>
              (lambda_log[i], mu[i], sigma_log, beta[0], beta[1]).
               derivative(0, 1, 0, 0, 0);
          else if(arg == 2)
            return g.lb_wo_constant<0, 0, 1, 0, 0>
              (lambda_log[i], mu[i], sigma_log, beta[0], beta[1]).
               derivative(0, 0, 1, 0, 0);
          else if(arg == 3)
            return g.lb_wo_constant<0, 0, 0, 1, 0>
              (lambda_log[i], mu[i], sigma_log, beta[0], beta[1]).
               derivative(0, 0, 0, 1, 0);
          else if(arg == 4)
            return g.lb_wo_constant<0, 0, 0, 0, 1>
              (lambda_log[i], mu[i], sigma_log, beta[0], beta[1]).
               derivative(0, 0, 0, 0, 1);

          throw std::invalid_argument("invalid 'arg'");
        };

        gr[0]    += func(2);
        gr[1]    += func(3);
        gr[2]    += func(4);
        gr[is++]  = func(0);
        gr[im++]  = func(1);
      }

      ++i;
    }
  }
};

/* wrap a T to an R list */
template<class T>
Rcpp::List wrap_return_obj(T const &x){
  return Rcpp::List::create(
    Rcpp::Named("lb") = x.lb, Rcpp::Named("gr") = std::move(x.gr));
}
}

//' @export
// [[Rcpp::export]]
Rcpp::List optim_VA_boost
  (arma::vec const &y, arma::vec const &x, double const sigma_log,
   arma::vec const &beta, arma::vec const &lambda_log, arma::vec const &mu,
   unsigned const grp_size, unsigned const order){
  size_t const n = y.n_elem,
        n_groups = n / grp_size;

  /* check arguments */
  if(n != x.n_elem or n < grp_size or n % grp_size > 0)
    throw std::invalid_argument("sizes differ");
  if(n_groups != mu.n_elem or n_groups != lambda_log.n_elem)
    throw std::invalid_argument("invalid group size");
  if(order > 1)
    throw std::invalid_argument("invalid order");
  if(beta.n_elem != 2)
    throw std::invalid_argument("invalid beta");

  /* setup object for each group */
  std::vector<lb_group> const lb_groups = ([&](){
    std::vector<lb_group> out;
    out.reserve(n_groups);

    double const *x_start = x.begin(), *y_start = y.begin();
    for(size_t i = 0; i < n_groups;
        ++i, y_start += grp_size, x_start += grp_size)
      out.emplace_back(grp_size, y_start, x_start);

    return out;
  })();

  /* compute and return */
  if(order == 0)
    return wrap_return_obj(comp_out<0>(
        lb_groups, lambda_log, mu, sigma_log, beta));

  return wrap_return_obj(comp_out<1>(
      lb_groups, lambda_log, mu, sigma_log, beta));
}
