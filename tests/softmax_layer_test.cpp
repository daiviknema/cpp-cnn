#define BOOST_TEST_MODULE SoftmaxLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/softmax_layer.hpp"

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  SoftmaxLayer s(3);
  arma::vec input(3, arma::fill::randn);
  arma::vec output;

  s.Forward(input, output);
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  SoftmaxLayer s(3);
  arma::vec input(3, arma::fill::randn);
  arma::vec output;

  s.Forward(input, output);

  arma::vec upstreamGradient = arma::ones(3);
  s.Backward(upstreamGradient);

  arma::vec gradWrtInput = s.getGradientWrtInput();

  arma::vec approxGradWrtInput = arma::zeros(3);

  double disturbance = 0.5e-5;
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    s.Forward(input, output);
    double l1 = arma::accu(output);
    input[i] -= 2.0*disturbance;
    s.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradWrtInput[i] = (l1-l2)/(2.0*disturbance);
    input[i] += disturbance;
  }
  BOOST_REQUIRE(arma::approx_equal(gradWrtInput,
                                   approxGradWrtInput,
                                   "absdiff",
                                   disturbance));
}
