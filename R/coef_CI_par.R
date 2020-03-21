#' Confidence Intervals and Estimates of Each Regression Coefficient - Parallelized
#'
#' This function takes in a list of linear regression coefficient estimates generated
#' by a Bag of Little Bootstraps procedure. Then, empirical confidence intervals and
#' point estimates of each coefficient are determined for each subsample. Afterwards,
#' the endpoints of all confidence intervals are averaged to form overall confidence
#' intervals, and point estimates are averaged to form overall estimates. It should be
#' noted that the confidence intervals are not multiple confidence intervals. For
#' Bonferroni-corrected confidence intervals, divide the desired value of alpha by the
#' number of regression coefficients. The difference between this function and coef_CI
#' is that this function uses parallel processing through furrr's future_map function.
#'
#' @param lrbs A linear_reg_bs or linear_reg_bs_par object containing BLB regression
#' coefficient estimates.
#' @param alpha The significance level. Default value is 0.05.
#' @return The overall confidence interval for each regression coefficient, along with
#' its overall estimate.
#' @export
coef_CI_par <- function(lrbs, alpha = 0.05) {
  coef <- lrbs$bootstrap_coefficient_estimates
  CIs <- future_map(coef, function(c) apply(c, 1, quantile,
                                            probs = c((alpha / 2), (1 - (alpha / 2)))))
  means <- future_map(coef, function(c) apply(c, 1, mean))
  CI <- reduce(CIs, `+`) / length(CIs)
  beta_hat <- reduce(means, `+`) / length(means)
  return(cbind(Lower_Bounds = CI[1,], Estimates = beta_hat, Upper_Bounds = CI[2,]))
}
