#define BOOST_TEST_MODULE ConvolutionLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "cnn.hpp"

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
  ConvolutionLayer c(
      5,  // Input height.
      5,  // Input width.
      3,  // Input depth.
      2,  // Filter height.
      3,  // Filter width.
      1,  // Horizontal stride.
      1,  // Vertical stride.
      3); // Number of filters.
}

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  arma::cube input(3, 3, 1, arma::fill::zeros);
  input.slice(0) = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

  arma::cube filter1(2, 2, 1, arma::fill::zeros);
  filter1.slice(0) = {{1, 0}, {0, 1}};

  arma::cube filter2(2, 2, 1, arma::fill::zeros);
  filter2.slice(0) = {{0, 1}, {1, 0}};

  std::vector<arma::cube> filters;
  filters.push_back(filter1);
  filters.push_back(filter2);

  ConvolutionLayer c(
      3,  // Input height.
      3,  // Input width.
      1,  // Input depth.
      2,  // Filter width.
      2,  // Filter depth.
      1,  // Horizontal stride.
      1,  // Vertical stride.
      2); // Number of filters.

  c.setFilters(filters);

  arma::cube output;
  c.Forward(input, output);
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  arma::cube input(3, 3, 1, arma::fill::zeros);
  input.slice(0) = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

  arma::cube filter1(2, 2, 1, arma::fill::zeros);
  filter1.slice(0) = {{1, 0}, {0, 1}};

  arma::cube filter2(2, 2, 1, arma::fill::zeros);
  filter2.slice(0) = {{0, 1}, {1, 0}};

  std::vector<arma::cube> filters;
  filters.push_back(filter1);
  filters.push_back(filter2);

  ConvolutionLayer c(
      3,  // Input height.
      3,  // Input width.
      1,  // Input depth.
      2,  // Filter width.
      2,  // Filter depth.
      1,  // Horizontal stride.
      1,  // Vertical stride.
      2); // Number of filters.

  c.setFilters(filters);

  arma::cube output;
  c.Forward(input, output);

  // For now, let the loss be the sum of all the output activations. Therefore,
  // the upstream gradient is all ones.
  arma::cube upstreamGradient(2, 2, 2, arma::fill::ones);

  c.Backward(upstreamGradient);

  arma::cube gradInput = c.getGradientWrtInput();

  std::vector<arma::cube> gradFilters = c.getGradientWrtFilters();

  // Now compute approximate gradients.
  double disturbance = 0.5e-5;

  output = arma::zeros(arma::size(output));
  arma::cube approxGradientWrtInput(arma::size(input), arma::fill::zeros);
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    c.Forward(input, output);
    double l1 = arma::accu(output);
    input[i] -= 2*disturbance;
    c.Forward(input, output);
    double l2 = arma::accu(output);
    approxGradientWrtInput[i] = (l1 - l2)/(2.0*disturbance);
  }

  std::cout << "Approx gradient wrt inputs:" << std::endl;
  std::cout << approxGradientWrtInput << std::endl;
}
