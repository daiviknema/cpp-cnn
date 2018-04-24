#define BOOST_TEST_MODULE MaxPoolingLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/max_pooling_layer.hpp"

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
  MaxPoolingLayer mp(
      7,  // Input height.
      5,  // Input width.
      4,  // Input depth.
      5,  // Pooling window height.
      3,  // Pooling window width.
      2,  // Vertical stride.
      2   // Horizontal stride.
      );
}

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  MaxPoolingLayer mp(
      7,  // Input height.
      5,  // Input width.
      4,  // Input depth.
      5,  // Pooling window height.
      3,  // Pooling window width.
      2,  // Vertical stride.
      2   // Horizontal stride.
      );

  arma::cube input(7, 5, 4, arma::fill::randn);
  arma::cube output;

  mp.Forward(input, output);
}


BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  MaxPoolingLayer mp(
      7,  // Input height.
      5,  // Input width.
      4,  // Input depth.
      5,  // Pooling window height.
      3,  // Pooling window width.
      2,  // Vertical stride.
      2   // Horizontal stride.
      );

  arma::cube input(7, 5, 4, arma::fill::randn);
  arma::cube output;

  mp.Forward(input, output);

  // Again, for now we loet the loss function be the sum of all output
  // activations. Therefore, the upstream gradient is all ones.
  arma::cube upstreamGradient = arma::ones(size(output));

  mp.Backward(upstreamGradient);

  arma::cube gradientWrtInput = mp.getGradientWrtInput();

  arma::cube approxGradientWrtInput = arma::zeros(arma::size(input));

  double disturbance = 0.5e-5;
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    mp.Forward(input, output);
    double l1 = arma::accu(output);
    input[i] -= 2.0*disturbance;
    mp.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradientWrtInput[i] = (l1-l2)/(2.0*disturbance);
    input[i] += disturbance;
  }

  BOOST_REQUIRE(arma::approx_equal(gradientWrtInput,
                                   approxGradientWrtInput,
                                   "absdiff",
                                   disturbance));
}
