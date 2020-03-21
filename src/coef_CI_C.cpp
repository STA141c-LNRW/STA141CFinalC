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
NumericMatrix coef_CI_C(List lrbs, double alpha){
  double lowerq = alpha/2;
  double upperq = 1 - alpha/2;
  List coefs = lrbs["bootstrap_coefficient_estimates"];
  NumericMatrix one = coefs[0];

  NumericMatrix prediction_intervals(one.nrow(),3);
  std::vector<double> lowerS(one.nrow(), 0);
  std::vector<double> upperS(one.nrow(), 0);
  std::vector<double> sumS(one.nrow(), 0);
  for (int i = 0; i < coefs.size(); i++){
    NumericMatrix s = coefs[i];
    NumericMatrix subset = transpose(s);
    for (int j = 0; j < subset.ncol(); j++){
      std::sort(subset.begin()+j*subset.nrow(),
                subset.begin()+(j+1)*subset.nrow());
      lowerS[j] += subset[j*subset.nrow()+subset.nrow()*lowerq];
      upperS[j] += subset[j*subset.nrow()+subset.nrow()*upperq];
      for (int k = 0; k < subset.nrow(); k++){
        sumS[j] += subset(k, j);
      }
    }
  }
  for (int i = 0; i < one.nrow(); i++){
    prediction_intervals(i, 0) = lowerS[i]/coefs.size();
    prediction_intervals(i, 2) = upperS[i]/coefs.size();
    prediction_intervals(i, 1) = sumS[i]/(coefs.size()*one.ncol());
  }
  rownames(prediction_intervals) = rownames(one);
  colnames(prediction_intervals) = CharacterVector(
  {"Lower_Bounds", "Estimates", "Upper_Bounds"});
  return prediction_intervals;
}
