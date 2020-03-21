//' This function takes in a dataframe of observations, split into explanatory variables
//' and response variable, and splits the data into a specified number of subsamples.
//' Then, each subsample is resampled a specified number of times. Afterwards, for each
//' resample, a linear regression model is fit, and the estimates for each regression
//' coefficient, as well as for the error variance. These estimates are returned to the
//' user, and they can be used to determine confidence intervals for the error variance
//' and each regression coefficient, as well as prediction intervals for new data. The
//' difference between this function and linear_reg_bs is that this function is written
//' in C++ instead of R for faster performance.
//' @param x A dataframe of the explanatory variables of all observations.
//' @param y A numeric vector of the response variable of all observations.
//' @param s The number of subsamples to split the data into. Default value is 10.
//' @param r The number of bootstrap samples to generate from each subsample. Default
//' value is 1000.
//' @return bootstrap_coefficient_estimates: The BLB estimates of all regression
//' coefficients. This list has an element for each subsample, and each element stores
//' the estimates for each bootstrap sample in a matrix.
//' @return bootstrap_s2_estimates: The BLB estimates of sigma-squared (error variance).
//' This list has an element for each subsample, and each element stores the estimates
//' for each bootstrap sample in a vector.
//' @export
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector s2_CI_C(List lrbs, double alpha){
  double lowerq = alpha/2;
  double upperq = 1 - alpha/2;
  List coefs = lrbs["bootstrap_s2_estimates"];
  double lower = 0.0;
  double upper = 0.0;
  double estimates = 0.0;
  for (int i = 0; i < coefs.size(); i++){
    NumericVector subset = coefs[i];
    double sum = 0;
    for (int i = 0; i < subset.size(); i++){
      sum += subset[i];
    }
    std::sort(subset.begin(), subset.end());
    lower += subset[subset.size()*lowerq];
    upper +=  subset[subset.size()*upperq];
    estimates += sum/subset.size();
  }
  lower /= coefs.size();
  estimates /= coefs.size();
  upper /= coefs.size();
  NumericVector prediction_intervals = NumericVector::create(
    _["Lower_Bound"] = lower,
    _["Estimate"] = estimates,
    _["Upper_Bound"] = upper
  );
  return prediction_intervals;
}
