#define BOOST_TEST_MODULE CrossEntropyLossLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/cross_entropy_loss_layer.hpp"

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  CrossEntropyLossLayer c(3);

  arma::vec predictedDistribution = {0.25, 0.25, 0.5};
  arma::vec actualDistribution1 = {1, 0, 0};
  arma::vec actualDistribution2 = {0, 0, 1};

  double loss1 = c.Forward(predictedDistribution, actualDistribution1);
  double loss2 = c.Forward(predictedDistribution, actualDistribution2);

  BOOST_REQUIRE(loss1 > loss2);
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  CrossEntropyLossLayer c(3);

  arma::vec predictedDistribution = {0.25, 0.25, 0.5};
  arma::vec actualDistribution = {0, 0, 1};

  double loss2 = c.Forward(predictedDistribution, actualDistribution);

  c.Backward();

  arma::vec gradientWrtPredictedDistribution =
      c.getGradientWrtPredictedDistribution();
  arma::vec approxGradient = arma::zeros(arma::size(predictedDistribution));

  double disturbance = 0.5e-5;
  for (size_t i=0; i<predictedDistribution.n_elem; i++)
  {
    predictedDistribution[i] += disturbance;
    double l1 = c.Forward(predictedDistribution, actualDistribution);
    predictedDistribution[i] -= 2.0*disturbance;
    double l2 = c.Forward(predictedDistribution, actualDistribution);
    approxGradient[i] = (l1-l2)/(2.0*disturbance);
    predictedDistribution[i] += disturbance;
  }

  BOOST_REQUIRE(arma::approx_equal(approxGradient,
                                   gradientWrtPredictedDistribution,
                                   "absdiff",
                                   disturbance));
}
