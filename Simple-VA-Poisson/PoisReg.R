library(TMB)
local({
  dir <- "Simple-VA-Poisson"
  if(!grepl(paste0(dir, "$"), getwd()))
    setwd(dir)
})

# compile code
stopifnot(compile("PoisReg.cpp") == 0)
dyn.load(dynlib("PoisReg"))

# simulate date
set.seed(42163306)
dat_org <- within(list(), {
  sigma <- .5
  beta <- c(2, 1.5)
  grp_size <- 5L
  n_groups <- 500L
  
  x <- runif(grp_size * n_groups, -1, 1)
  grp <- rep(1:n_groups, each = grp_size)
  y <- rpois(grp_size * n_groups, lambda = exp(
    beta[1] + beta[2] * x + rnorm(n_groups, sd = sigma)[grp]))
})

#####
# setup functions to use to optimize variational parameters and parameters 
# in the model
params <- with(dat_org, list(
  sigma_log = 0,
  beta = c(log(mean(y)), 0),
  mu = rep(0, n_groups),
  lambda_log = rep(0, n_groups)))
ad_func <- MakeADFun(
  data = dat_org[c("y", "x", "grp_size")], parameters = params, DLL = "PoisReg",
  silent = TRUE)

# function that returns a function to optimize a subset of the parameters 
get_top_func <- function(params_update, reltol = 1e-5){
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

#####
# use function with two-step procedure where we first optimize over variational 
# parameters and then over model parameters
optim_two_step <- function(obj, maxit = 1000L, eps = 1e-8, verbose = TRUE){
  opt_variational <- get_top_func(c("mu", "lambda_log"), reltol = eps)
  opt_params <- get_top_func(c("sigma_log", "beta"), reltol = eps)
  
  lls <- rep(NA_real_, maxit)
  ll_old <- -.Machine$double.xmax
  for(i in 1:maxit){
    obj <- opt_variational(obj, verbose = FALSE)
    obj <- opt_params(obj, verbose = verbose && (i - 1L) %% 10 == 0)
    
    val <- obj$value
    lls[i] <- val
    if(abs((val - ll_old) / ll_old) < eps)
      break
    ll_old <- val
  
  }
  
  obj$n_it <- i
  obj$lls <- lls[1:i]
  obj
  
}

two_est <- optim_two_step(ad_func)

plot(two_est$lls) # lower bounds vs iterations
exp(two_est$par["sigma_log"]) # estimated std. dev.

#####
# just maximize directly 
ad_func$control <- list(fnscale = -1, maxit = 1000L)
one_est <- do.call(optim, ad_func)

# difference in parameter estimates
two_est$par[1:3] - one_est$par[1:3]
one_est$par[1:3]
exp(one_est$par["sigma_log"])

#####
# compare w/ adaptive gaussian quadrature
library(lme4)
lme4_fit <- glmer(y ~ x + (1 | grp), dat_org, poisson(), nAGQ = 10)
summary(lme4_fit)

# compare log-likelihood and lower bound
logLik(lme4_fit)
one_est$value
