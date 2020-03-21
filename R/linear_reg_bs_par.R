#' Linear Regression Using Bag of Little Bootstraps - Parallelized
#'
#' This function takes in a dataframe of observations, split into explanatory variables
#' and response variable, and splits the data into a specified number of subsamples.
#' Then, each subsample is resampled a specified number of times. Then, for each
#' resample, a linear regression model is fit, and the estimates for each regression
#' coefficient, as well as for the error variance. These estimates are returned to the
#' user, and they can be used to determine confidence intervals for the error variance
#' and each regression coefficient, and prediction intervals for new data. The
#' difference between this function and linear_reg_bs is that this function uses
#' parallel processing through furrr's future_map function.
#'
#' @param x A dataframe of the explanatory variables of all observations.
#' @param y A numeric vector of the response variable of all observations.
#' @param s The number of subsamples to split the data into. Default value is 10.
#' @param r The number of bootstrap samples to generate from each subsample. Default
#' value is 1000.
#' @return bootstrap_coefficient_estimates: The BLB estimates of all regression
#' coefficients. This list has an element for each subsample, and each element stores
#' the estimates for each bootstrap sample in a matrix.
#' @return bootstrap_s2_estimates: The BLB estimates of sigma-squared (error variance).
#' This list has an element for each subsample, and each element stores the estimates
#' for each bootstrap sample in a vector.
#' @export
linear_reg_bs_par <- function(x, y, s = 10, r = 1000) {
  n <- dim(x)[1]
  p <- dim(x)[2] + 1
  x1 <- cbind(Intercept = rep(1, n), x)
  sample_indices <- sample(n)
  samples <- sample(s)
  x_samples <- split(x1[sample_indices,], samples)
  y_samples <- split(y[sample_indices], samples)
  bs_coefs_s2 <- future_map(1:s, function(i) {
    n_sub <- length(y_samples[[i]])
    subset <- data.frame(x_samples[[i]], y_samples[[i]])
    sample_coefs_s2 <- future_map(1:r, function(j) {
      freqs <- rmultinom(1, n, rep(1, n_sub))
      resamp = subset[rep(seq_len(nrow(subset)), freqs),]
      x_resamp <- as.matrix(resamp[, 1:p])
      y_resamp <- as.matrix(resamp[, p+1])
      coefs <- solve(t(x_resamp) %*% x_resamp) %*% t(x_resamp) %*% y_resamp
      fv <- x_resamp %*% coefs
      res <- y_resamp - fv
      s2 <- sum(res^2) / (n_sub - p)
      list(sample_coefs = coefs, sample_s2 = s2)
    })
    sample_s2 <- future_map(sample_coefs_s2, function(scs2) scs2$sample_s2)
    sample_coefs <- future_map(sample_coefs_s2, function(scs2) scs2$sample_coefs)
    list(bs_coefs = sample_coefs, bs_s2 = sample_s2)
  })
  bs_coefs <- future_map(bs_coefs_s2, function(bscs2) bscs2$bs_coefs)
  bs_coefs <- future_map(bs_coefs, function(sample) matrix(unlist(sample), nrow = 15))
  bs_s2 <- future_map(bs_coefs_s2, function(bscs2) bscs2$bs_s2)
  bs_s2 <- future_map(bs_s2, function(sample) unlist(sample))
  return(list(bootstrap_coefficient_estimates = bs_coefs,
              bootstrap_s2_estimates = bs_s2))
}
