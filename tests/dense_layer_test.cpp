#define BOOST_TEST_MODULE DenseLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/dense_layer.hpp"

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
  DenseLayer d(
      5,    // Input height.
      5,    // Input width.
      3,    // Input depth.
      10);  // Number of outputs.
}

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  DenseLayer d(
      5,    // Input height.
      5,    // Input width.
      3,    // Input depth.
      10);  // Number of outputs.

  arma::cube input(5, 5, 3, arma::fill::randn);
  arma::vec output;

  d.Forward(input, output);
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  DenseLayer d(
      5,    // Input height.
      5,    // Input width.
      3,    // Input depth.
      10);  // Number of outputs.

  arma::cube input(5, 5, 3, arma::fill::randn);
  arma::mat weights = d.getWeights();
  arma::vec output;

  d.Forward(input, output);

  // Again, for now we loet the loss function be the sum of all output
  // activations. Therefore, the upstream gradient is all ones.
  arma::vec upstreamGradient = arma::ones(size(output));

  d.Backward(upstreamGradient);

  arma::cube gradWrtInput = d.getGradientWrtInput();
  arma::mat gradWrtWeights = d.getGradientWrtWeights();

  arma::cube approxGradWrtInput = arma::zeros(size(input));
  arma::mat approxGradWrtWeights = arma::zeros(size(weights));

  double disturbance = 0.5e-5;
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    d.Forward(input, output);
    double l1 = arma::accu(output);
    input[i] -= 2.0*disturbance;
    d.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradWrtInput[i] = (l1-l2)/(2.0*disturbance);
    input[i] += disturbance;
  }

  BOOST_REQUIRE(arma::approx_equal(gradWrtInput,
                                   approxGradWrtInput,
                                   "absdiff",
                                   disturbance));

  for (size_t i=0; i<weights.n_elem; i++)
  {
    weights[i] += disturbance;
    d.setWeights(weights);
    d.Forward(input, output);
    double l1 = arma::accu(output);
    weights[i] -= 2.0*disturbance;
    d.setWeights(weights);
    d.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradWrtWeights[i] = (l1-l2)/(2.0*disturbance);
    weights[i] += disturbance;
    d.setWeights(weights);
  }

  BOOST_REQUIRE(arma::approx_equal(gradWrtWeights,
                                   approxGradWrtWeights,
                                   "absdiff",
                                   disturbance));
}
