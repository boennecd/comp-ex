VA with CppAD for GSM
================
Benjamin Christoffersen
27 September, 2019

This package is made to

-   show implementations of Laplace approximations and Gaussian quadrature using `CppAD` through `TMB`.
-   show implementations of variational approximation in generalized survival models (GSM).
-   how to use `TMB` to get extra operators for `CppAD` and make `CppAD` work with `Eigen`.
-   use `TMB` and `Rcpp` together and have multiple C++ functions which can
    be used for R without one function with a switch statement.

The latter may be advantageous later as we can work more solely in C++. You can contrast the solution here with the discussion made in [this post](https://github.com/kaskr/adcomp/issues/247#issuecomment-475002639). We start by using a temporary directory as our working directory throughout the rest of the code

``` r
old_dir <- getwd()
dir_use <- tempdir()
knitr::opts_knit$set(root.dir = dir_use)
```

We also load a few libraries

``` r
library(GSMVAEx)
library(TMB)
```

log-log GSM
-----------

So far, we just compute the gradient of log-log GSM in `C++` and compare it with using `TMB` from R. Here is the `loglogTMBEx.cpp` file which we will use in `R`

``` cpp
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
```

We make a copy of the file in the temporary directory and compile the code.

``` r
stopifnot(
  file.copy(file.path(old_dir, "README", "loglogTMBEx.cpp"),
          "loglogTMBEx.cpp"),
  compile("loglogTMBEx.cpp") == 0)
```

    ## Note: Using Makevars in /home/boennecd/.R/Makevars

``` r
dyn.load(dynlib("loglogTMBEx"))
```

Then we have a bit of R code which we need in order to simulate the data and which we use in subsequent code (this is some code I received and there seems to be a bug in it)

``` r
require(splines)
```

    ## Loading required package: splines

``` r
set.seed(12345)
x <- seq(0, 1, length=1e4)
time <- rweibull(length(x), 1, exp(.1+0.1*x))
censor <- runif(length(x), 0, 5)
event <- time < censor
tobs <- pmin(censor, time)
formula <- ~ x + ns(log(tobs) , df = 1)

# get the terms object we want
trms_use <- local({
  mf <- model.frame(formula, data.frame(tobs = tobs, x = x)[event, ])
  terms(mf)
})

getXD <- function(trms, newdata, time, eps = 1e-5) {
  stopifnot(inherits(trms, "terms"))
  newdata[[time]] <- newdata[[time]] + eps
  upper <- model.matrix(trms, newdata)
  newdata[[time]] <- newdata[[time]] - eps
  (upper - model.matrix(trms, newdata)) / 2 / eps
}

dat <- data.frame(tobs, x)
X <- model.matrix(trms_use, data.frame(tobs, x))
XD <- getXD(trms_use, dat, time = "tobs")

MakeADFun <- function(...) {
  newobj <- TMB::MakeADFun(...)
  newobj$call <- match.call()
  newobj
}
```

We compare the computation of the gradient in C++ in this package with the one using the `TMB` interface from `R`

``` r
local({
  # assign functions to do the recording and compute the gradient
  tmb <- function(){
    f <- MakeADFun(
      data = list(X = X, XD = XD, tobs = tobs, event = as.double(event),
                  eps = 1e-6, kappa = 1.0),
      parameters = list(beta = c(-5, 0, 1)), method = "nlminb",
      DLL = "loglogTMBEx", silent = TRUE)
    
    structure(f$gr(c(-5, 0, 1)), func = f)
  }
  imp <- function()
    GSMVAEx::loglog(tobs = tobs, event = as.double(event), XD = XD, X = X, 
                    eps = 1e-6, kappa = 1.0, beta = c(-5, 0, 1))
  
  # we get the same gradient
  stopifnot(isTRUE(all.equal(c(tmb()), c(imp()))))
  
  cat("The computation time are comparable\n")
  print(microbenchmark::microbenchmark(TBM = tmb(), Implementation = imp(), 
                                       times = 100))
  
  cat("\nA large part is the \"tape recording\"\n")
  f <- attr(tmb(), "func")
  microbenchmark::microbenchmark(
    `Full cost`          = tmb(),
    `Non recording cost` = f$gr(c(-5, 0, 1)), times = 100)
})
```

    ## The computation time are comparable
    ## Unit: milliseconds
    ##            expr   min    lq  mean median    uq   max neval
    ##             TBM 14.58 15.10 16.33  15.43 15.82 90.66   100
    ##  Implementation 10.94 11.44 12.21  11.93 12.55 23.39   100
    ## 
    ## A large part is the "tape recording"

    ## Unit: milliseconds
    ##                expr    min     lq   mean median     uq    max neval
    ##           Full cost 14.690 15.034 16.121 15.318 15.678 86.323   100
    ##  Non recording cost  1.562  1.596  1.645  1.628  1.672  2.035   100

Of course, one of the nice things about `MakeADFun` is that we do not have to do the recording over and over again and save a lot of computation time. We can similarly save a pointer to a C++ object in our implementation.

### Optimization in C++

We can also compare computation of the parameters. This is done in the C++ code using the `nlopt` library through the `nloptr` package

``` r
local({
  # assign functions to do the recording and compute the gradient
  tmb <- function(){
    f <- MakeADFun(
      data = list(X = X, XD = XD, tobs = tobs, event = as.double(event),
                  eps = 1e-6, kappa = 1.0),
        parameters = list(beta = c(-5, 0, 1)), method = "nlminb",
      DLL = "loglogTMBEx", silent = TRUE)
    opt <- optim(c(-5, 0, 1), fn = f$fn, gr = f$gr)
    opt[c("par", "value")]
  }
  imp <- function()
    GSMVAEx::loglog_opt(
      tobs = tobs, event = as.double(event), XD = XD, X = X, eps = 1e-6, 
      kappa = 1.0, beta = c(-5, 0, 1))
  
  # we get the same result
  stopifnot(isTRUE(all.equal(tmb(), imp(), tolerance = 1e-4)))
  
  cat("The computation time are comparable\n")
  microbenchmark::microbenchmark(TBM = tmb(), Implementation = imp(), 
                                 times = 10)
})
```

    ## The computation time are comparable

    ## Unit: milliseconds
    ##            expr   min    lq  mean median     uq    max neval
    ##             TBM 95.69 96.30 98.44  98.91 100.27 101.35    10
    ##  Implementation 66.19 66.73 67.11  67.09  67.48  68.48    10

Finally, notice that we have called more than one C++ function form R from
the `src/R-interface.cpp` file shown below

``` cpp
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
```
