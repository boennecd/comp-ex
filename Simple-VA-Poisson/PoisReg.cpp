#include <TMB.hpp>
#include <stdexcept>

template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(y);
  DATA_VECTOR(x);
  DATA_INTEGER(grp_size);
  
  PARAMETER(sigma_log);
  PARAMETER_VECTOR(beta);
  PARAMETER_VECTOR(mu);
  PARAMETER_VECTOR(lambda_log);
  
  Type sigma = exp(sigma_log);
  vector<Type> lambda = exp(lambda_log);
  
  /* check args. The code currently crash with C++ exceptions though... */
  std::size_t const n = y.size(), n_groups = n / grp_size;
  if(n != (unsigned)x.size() or n < (unsigned)grp_size or n % grp_size > 0)
    throw std::invalid_argument("sizes differ");
  if(n_groups != (unsigned)mu.size() or n_groups != (unsigned)lambda.size())
    throw std::invalid_argument("invalid group size");
  
  /* compute lower bound */
  Type lb = Type(0);
  {
    Type const sigma_sq = sigma * sigma, 
                   half(.5);
    lb += Type(n_groups) * half * (1. - log(sigma_sq));
    
    std::size_t j = 0;
    for(std::size_t i = 0; i < n_groups; ++i){
      Type const 
        lambda_half = lambda[i] * half,
               term = mu[i] + lambda_half;
      lb += half * (log(lambda[i]) - (mu[i] * mu[i] + lambda[i]) / sigma_sq);
      
      std::size_t const n_j = (i + 1) * grp_size;
      for(; j < n_j; ++j)
        lb += dpois(y[j], exp(beta[0] + beta[1] * x[j] + term), true) - 
            y[j] * lambda_half;
    }
  }
  
  return lb;
}
