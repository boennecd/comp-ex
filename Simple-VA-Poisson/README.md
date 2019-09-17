Simple Poisson Mixed Model
================
Benjamin Christoffersen
17 September, 2019

-   [Simulate Data](#simulate-data)
-   [Using TMB (CppAD)](#using-tmb-cppad)
    -   [Two-Step Procedure](#two-step-procedure)
    -   [Optimize with VA](#optimize-with-va)
-   [Compare with Adaptive Gauss-Hermite Quadrature](#compare-with-adaptive-gauss-hermite-quadrature)
-   [References](#references)

This directory contains an example of the Poisson mixed model described in Ormerod and Wand (2010).

Simulate Data
-------------

We simulate data as follows

``` r
set.seed(42163306)
dat_org <- within(list(), {
  sigma <- 1.5      # std of random intercept term
  beta <- c(2, 1.5) # intercept and slope
  grp_size <- 5L    # number of individuals per group
  n_groups <- 50L   # number of groups
  
  # simulate individual specific covariates and observed outcomes
  x <- runif(grp_size * n_groups, -1, 1)
  grp <- rep(1:n_groups, each = grp_size)
  y <- rpois(grp_size * n_groups, lambda = exp(
    beta[1] + beta[2] * x + rnorm(n_groups, sd = sigma)[grp]))
})
```

We assume equal group sizes for simplicity.

Using TMB (CppAD)
-----------------

The following sections uses the `TMB` package which uses the `CppAD` C++ library to perform automatic differentiation. First, we compile the C++ file we need

``` r
library(TMB)
stopifnot(compile("PoisReg.cpp") == 0)
dyn.load(dynlib("PoisReg"))
```

Here is the `PoisReg.cpp` file

``` cpp
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
  
  Type const sigma = exp(sigma_log);
  vector<Type> const lambda = exp(lambda_log);
  
  /* check args. The code currently crash with C++ exceptions though... */
  std::size_t const n = y.size(), n_groups = n / grp_size;
  if(n != (unsigned)x.size() or n < (unsigned)grp_size or n % grp_size > 0)
    throw std::invalid_argument("sizes differ");
  if(n_groups != (unsigned)mu.size() or n_groups != (unsigned)lambda.size())
    throw std::invalid_argument("invalid group size");
  
  /* compute lower bound */
  Type lb(0);
  {
    Type const sigma_sq = sigma * sigma;
    lb += n_groups * .5 * (1. - log(sigma_sq));
    
    std::size_t j = 0;
    for(std::size_t i = 0; i < n_groups; ++i){
      Type const 
        lambda_half = lambda[i] * .5,
               term = mu[i] + lambda_half;
      lb += .5 * (log(lambda[i]) - (mu[i] * mu[i] + lambda[i]) / sigma_sq);
      
      std::size_t const n_j = (i + 1) * grp_size;
      for(; j < n_j; ++j)
        lb += dpois(y[j], exp(beta[0] + beta[1] * x[j] + term), true) - 
            y[j] * lambda_half;
    }
  }
  
  return lb;
}
```

### Two-Step Procedure

We define a function which returns a function which optimizes a subset of the parameters. This will allow us to iterate between optimizing the variational parameters and the model parameters.

``` r
get_opt_func <- function(params_update, reltol = 1e-5){
  eval(bquote(function(obj, verbose = FALSE){
    # setup the objects we need
    par_all <- obj$par
    idx_update <- names(par_all) %in% .(params_update)
    par <- par_all[idx_update]
    fn <- obj$fn
    gr <- obj$gr
    
    # setup functions we need
    fn_use <- function(x, ...){
      z <- par_all
      z[idx_update] <- x
      fn(z)
    }
    gr_use <- function(x, ...){
      z <- par_all
      z[idx_update] <- x
      drop(gr(z))[idx_update]
    }
    
    # optimize
    opt_res <- optim(par, fn = fn_use, gr = gr_use, method = obj$method, 
                     control = list(fnscale = -1, maxit = 1000L, 
                                    reltol = .(reltol)))
    if(opt_res$convergence > 0)
      stop(sprintf("optim failed with code %d\n", opt_res$convergence))
    obj$par[idx_update] <- opt_res$par
    obj$value <- opt_res$value
    
    # print and return
    if(verbose){
      cat(sprintf("\nLower bound is %.4f. Parameters estimates are:\n", 
                  obj$value))
      print(obj$par[idx_update])
    }
    
    obj
  }))
}
```

### Optimize with VA

Next, we use the two-step procedure. First, we get an object which gives a functions to evaluate the lower bound, the gradient of the lower bound, etc.

``` r
# starting values
params <- with(dat_org, list(
   sigma_log = 0,
        beta = c(log(mean(y)), 0),
          mu = rep(0, n_groups),
  lambda_log = rep(0, n_groups)))

# assign object with lower bound function, gradient, etc. 
ad_func <- MakeADFun(
  data = dat_org[c("y", "x", "grp_size")], parameters = params, DLL = "PoisReg",
  silent = TRUE)
```

Then we use the two-step procedure

``` r
# assign function to perform two-step procedure
optim_two_step <- function(obj, maxit = 1000L, eps = 1e-8, verbose = FALSE){
  opt_variational <- get_opt_func(c("mu", "lambda_log"), reltol = eps)
  opt_params <- get_opt_func(c("sigma_log", "beta"), reltol = eps)
  
  lbs <- rep(NA_real_, maxit)
  lb_old <- -.Machine$double.xmax
  for(i in 1:maxit){
    obj <- opt_variational(obj, verbose = FALSE)
    obj <- opt_params(obj, verbose = verbose && (i - 1L) %% 10 == 0)
    
    val <- obj$value
    lbs[i] <- val
    if(abs((val - lb_old) / lb_old) < eps)
      break
    lb_old <- val
  
  }
  
  obj$n_it <- i
  obj$lbs <- lbs[1:i]
  obj
  
}

# optimize parameters
two_est <- optim_two_step(ad_func)

# plot of lower bound versus iteration index
plot(two_est$lbs, xlab = "iteration", ylab = "lower bound", type = "l")
```

![](figures/README-tmb_use_two_step-1.png)

``` r
# parameter estimates
two_est$par[names(two_est$par) == "beta"]
```

    ##  beta  beta 
    ## 1.738 1.470

``` r
exp(two_est$par["sigma_log"])
```

    ## sigma_log 
    ##     1.281

We can compare this to optimizing over all parameters

``` r
ad_func$control <- list(fnscale = -1, maxit = 1000L)
one_est <- do.call(optim, ad_func)

# compare results
two_est$par[1:3] - one_est$par[1:3]
```

    ##  sigma_log       beta       beta 
    ## -0.0003536  0.0259633  0.0000593

Compare with Adaptive Gauss-Hermite Quadrature
----------------------------------------------

We can compare the result with using adaptive Gauss-Hermite quadrature

``` r
library(lme4)
```

    ## Warning: package 'lme4' was built under R version 3.5.3

    ## Loading required package: Matrix

``` r
lme4_fit <- glmer(y ~ x + (1 | grp), dat_org, poisson(), nAGQ = 10)
(lme4_fit_sum <- summary(lme4_fit))
```

    ## Generalized linear mixed model fit by maximum likelihood (Adaptive
    ##   Gauss-Hermite Quadrature, nAGQ = 10) [glmerMod]
    ##  Family: poisson  ( log )
    ## Formula: y ~ x + (1 | grp)
    ##    Data: dat_org
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##    449.0    459.6   -221.5    443.0      247 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.1063 -0.6417 -0.0501  0.5902  2.2573 
    ## 
    ## Random effects:
    ##  Groups Name        Variance Std.Dev.
    ##  grp    (Intercept) 1.65     1.28    
    ## Number of obs: 250, groups:  grp, 50
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   1.7113     0.1852    9.24   <2e-16 ***
    ## x             1.4696     0.0371   39.61   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##   (Intr)
    ## x -0.075

``` r
# compare log-likelihood and lower bound
logLik(lme4_fit)
```

    ## 'log Lik.' -221.5 (df=3)

``` r
one_est$value
```

    ## [1] -653.1

We can compare the estimated standard errors

``` r
# from VA
local({
  hess <- ad_func$he(one_est$par)
  out <- sqrt(diag(solve(-hess)))
  names(out) <- names(one_est$par)
  out[1:3]
})
```

    ## sigma_log      beta      beta 
    ##    0.1059    0.1850    0.0371

``` r
# standard errors for beta
sqrt(diag(lme4_fit_sum$vcov)) # lme4 betas std
```

    ## [1] 0.1852 0.0371

References
----------

Ormerod, J. T., and M. P. Wand. 2010. “Explaining Variational Approximations.” *The American Statistician* 64 (2). Taylor & Francis: 140–53. doi:[10.1198/tast.2010.09058](https://doi.org/10.1198/tast.2010.09058).
