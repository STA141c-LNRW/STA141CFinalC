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
NumericMatrix PI_C(List lrbs, DataFrame x, double alpha){
  double lowerq = alpha/2;
  double upperq = 1 - alpha/2;
  List coefs = lrbs["bootstrap_coefficient_estimates"];
  NumericMatrix temp = coefs[0];
  int replications = temp.ncol();
  int n = x.nrows();
  int p = x.size() + 1;
  arma::mat z(n,p);
  std::fill(z.begin(), z.begin() + n, 1);
  for (int i = 1; i < p; i++){
    NumericVector temp = x[i-1];
    std::copy(temp.begin(), temp.end(), z.begin() + i*n);
  }
  arma::cube preds(replications, coefs.size(), x.nrows(), arma::fill::zeros);
  NumericMatrix prediction_intervals(x.nrows(), 3);
  for (int i = 0; i < coefs.size(); i++){
    NumericMatrix sub_coefs = coefs[i];
    for (int j = 0; j < replications; j++){
      for (int k = 0; k < n; k++){
        for (int m = 0; m < p; m++){
          preds(j,i,k) += z(k,m)*sub_coefs(m,j);
        }
      }
    }
  }
  for (int i = 0; i < x.nrows(); i++){
    double lowerE = 0.0;
    double estimatesE = 0.0;
    double upperE = 0.0;
    for (int k = 0; k < coefs.size(); k++){
        double sum = 0;
        std::vector<double> subset(replications);
        for (int j = 0; j < replications; j++){
          subset[j] = preds(j,k,i);
          sum += subset[j];
        }
        std::sort(subset.begin(), subset.end());
        lowerE += subset[subset.size()*lowerq];
        upperE +=  subset[subset.size()*upperq];
        estimatesE += sum/subset.size();
    }
    prediction_intervals(i, 0) = lowerE/coefs.size();
    prediction_intervals(i, 1) = estimatesE/coefs.size();
    prediction_intervals(i, 2) = upperE/coefs.size();
  }
  colnames(prediction_intervals) = CharacterVector(
    {"Lower_Bounds", "Estimates", "Upper_Bounds"});
  return prediction_intervals;
}

