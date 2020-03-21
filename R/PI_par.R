#' Prediction Intervals and Estimates for New Data - Parallelized
#'
#' This function takes in a list of linear regression coefficient estimates generated
#' by a Bag of Little Bootstraps procedure, and a dataframe of observations without the
#' response variable. The response variable for each observation is predicted using
#' each vector of coefficient estimates for each sample. Then, empirical prediction
#' intervals and point estimates for the response variable of each observation are
#' determined for each sample. Afterwards, the endpoints of all intervals are averaged
#' to form overall prediction intervals, and point estimates are averaged to form
#' overall predictions. It should be noted that the prediction intervals are not
#' multiple prediction intervals. For Bonferroni-corrected prediction intervals, divide
#' the desired value of alpha by the number of observations. The difference between
#' this function and PI is that this function uses parallel processing through furrr's
#' future_map function.
#'
#' @param lrbs A linear_reg_bs or linear_reg_bs_par object containing BLB regression
#' coefficient estimates.
#' @param x A dataframe of the explanatory variables of unseen observations.
#' @param alpha The significance level. Default value is 0.05.
#' @return The prediction intervals and estimates for the response variable of each
#' unseen observation.
#' @export
PI_par <- function(lrbs, x, alpha = 0.05) {
  coefs <- lrbs$bootstrap_coefficient_estimates
  x1 <- as.matrix(cbind(Intercept = 1, x))
  preds <- future_map(1:length(coefs), function(i) {
    future_map(1:dim(coefs[[i]])[2], function(j) x1 %*% coefs[[i]][, j])
  })
  preds <- future_map(preds, function(sample) matrix(unlist(sample),nrow = dim(x1)[1]))
  PIs <- future_map(preds, function(p) apply(p, 1, quantile,
                                             probs = c((alpha/2), (1 - (alpha/2)))))
  fits <- future_map(preds, function(p) apply(p, 1, mean))
  PI <- reduce(PIs, `+`) / length(PIs)
  fit <- reduce(fits, `+`) / length(fits)
  return(cbind(Lower_Bounds = PI[1,], Estimates = fit, Upper_Bounds = PI[2,]))
}
