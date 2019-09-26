VA with CppAD for GSM
================
Benjamin Christoffersen
26 September, 2019

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
file.copy(file.path(old_dir, "README", "loglogTMBEx.cpp"),
          "loglogTMBEx.cpp")
```

    ## [1] TRUE

``` r
stopifnot(compile("loglogTMBEx.cpp") == 0)
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

getXD <- function(formula, data, newdata, time, eps = 1e-5) {
    newdata[[time]] <- newdata[[time]] + eps
    upper <- rstpm2:::predict.formula(formula, data, newdata)
    newdata[[time]] <- newdata[[time]]-2*eps
    (upper-rstpm2:::predict.formula(formula,data,newdata))/2/eps
}

# TODO: something seems to go wrong here? At least the dimension of beta 
# which we assign later is incorrect in the original code
X <- rstpm2:::predict.formula(
  formula, data.frame(tobs = tobs[event], x = x[event]), data.frame(tobs, x))
```

    ## Warning in model.matrix.default(mt, mfnew, contrasts = contrasts): non-list
    ## contrasts argument ignored

``` r
head(X) # why is there no `ns(log(tobs) , df = 1)` column?
```

    ##   (Intercept)         x
    ## 1           1 0.0000000
    ## 2           1 0.0001000
    ## 3           1 0.0002000
    ## 4           1 0.0003000
    ## 5           1 0.0004000
    ## 6           1 0.0005001

``` r
XD <- getXD(formula, data.frame(tobs = tobs[event], x = x[event]), 
            data.frame(tobs, x), time = "tobs")
```

    ## Warning in model.matrix.default(mt, mfnew, contrasts = contrasts): non-list
    ## contrasts argument ignored

    ## Warning in model.matrix.default(mt, mfnew, contrasts = contrasts): non-list
    ## contrasts argument ignored

``` r
MakeADFun <- function(...) {
  newobj <- TMB::MakeADFun(...)
  newobj$call <- match.call()
  newobj
}
```

Finally, we compare the computation of the gradient in C++ in this package with the one using the `TMB` interface from `R`

``` r
local({
  # assign functions to do the recording and compute the gradient
  tmb <- function(){
    f = MakeADFun(
      data = list(X = X, XD = XD, tobs = tobs, event = as.double(event),
                  eps = 1e-6, kappa = 1.0),
      # TODO: original code
      # parameters = list(beta = c(-5, 0, 1)), method = "nlminb",
        parameters = list(beta = c(-5, 0   )), method = "nlminb",
      DLL = "loglogTMBEx", silent = TRUE)
    f$gr(c(-5, 0))
  }
  imp <- function()
    GSMVAEx::loglog(tobs = tobs, event = as.double(event), XD = XD, X = X, 
                    eps = 1e-6, kappa = 1.0, beta = c(-5, 0))
  
  # we get the same gradient
  stopifnot(all.equal(drop(tmb()), drop(imp())))
  
  # the speed is the same
  microbenchmark::microbenchmark(TBM = tmb(), Implementation = imp(), 
                                 times = 100)
})
```

    ## Unit: milliseconds
    ##            expr   min    lq  mean median    uq    max neval
    ##             TBM 7.805 8.237 9.507  8.373 8.686 99.754   100
    ##  Implementation 4.656 4.921 5.235  5.122 5.356  8.082   100

Of course, one of the nice things about `MakeADFun` is that we do not have to do the recording over and over again and save a lot of computation time. We can similarly save a pointer to a C++ object in our implementation. Finally, to make the point that we can call multiple different C++ functions from R then we make a call to the `hello_world_func` function in `src/R-interface.cpp` file shown below

``` cpp
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
```

here

``` r
hello_world_func()
```

    ## Hello world
