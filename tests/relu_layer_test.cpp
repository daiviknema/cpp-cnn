#define BOOST_TEST_MODULE ReLULayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/relu_layer.hpp"

BOOST_AUTO_TEST_CASE(ContructorTest)
{
  ReLULayer r(5, 7, 3);
}

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  ReLULayer r(5, 7, 3);
  arma::cube input(5, 7, 3, arma::fill::randn);
  arma::cube output;

  r.Forward(input, output);
  BOOST_REQUIRE(arma::size(input) == arma::size(output));
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  ReLULayer r(5, 7, 3);
  arma::cube input(5, 7, 3, arma::fill::randn);
  arma::cube output;

  r.Forward(input, output);
  r.Backward(arma::ones(arma::size(output)));

  arma::cube gradientWrtInput = r.getGradientWrtInput();

  arma::cube approxGradientWrtInput = arma::zeros(arma::size(input));

  double disturbance = 0.5e-5;
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    r.Forward(input, output);
    double l1 = arma::accu(output);
    input[i] -= 2.0*disturbance;
    r.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradientWrtInput[i] = (l1-l2)/(2.0*disturbance);
    input[i] += disturbance;
  }
  BOOST_REQUIRE(arma::approx_equal(gradientWrtInput,
                                   approxGradientWrtInput,
                                   "absdiff",
                                   disturbance));
}
