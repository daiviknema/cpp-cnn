#define BOOST_TEST_MODULE ConvolutionLayerTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/convolution_layer.hpp"

#define DEBUG false
#define DEBUG_PREFIX "[CONV LAYER TESTS ]\t"

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
    input[i] += disturbance;
  }

#if DEBUG
  std::cout
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl
      << DEBUG_PREFIX << "BACKWARD PASS TEST (BackwardPassTest) DEBUG OUTPUT"
      << std::endl
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl;
  std::cout << DEBUG_PREFIX << "Approx gradient wrt inputs:" << std::endl;
  for (size_t s=0; s<approxGradientWrtInput.n_slices; s++)
  {
    std::cout << DEBUG_PREFIX << "Slice #" << s << std::endl;
    for (size_t r=0; r<approxGradientWrtInput.slice(s).n_rows; r++)
      std::cout << DEBUG_PREFIX << approxGradientWrtInput.slice(s).row(r);
  }
#endif
}

BOOST_AUTO_TEST_CASE(BackwardPassBigTest)
{
  // Input is 7 rows, 11 cols, and 3 slices.
  arma::cube input(7, 11, 3, arma::fill::randn);

  ConvolutionLayer c(
      7,  // Input height.
      11,  // Input width.
      3,  // Input depth.
      3,  // Filter height.
      5,  // Filter width.
      2,  // Horizontal stride.
      2,  // Vertical stride.
      2); // Number of filters.

  arma::cube output;
  c.Forward(input, output);

  // For now, let the loss be the sum of all the output activations. Therefore,
  // the upstream gradient is all ones.
  arma::cube upstreamGradient(3, 4, 2, arma::fill::ones);

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
    input[i] += disturbance;
  }

#if DEBUG
  std::cout
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl
      << DEBUG_PREFIX << "BACKWARD PASS TEST (BackwardPassBigTest) DEBUG OUTPUT"
      << std::endl
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl;
  std::cout << DEBUG_PREFIX << "Approx gradient wrt inputs:" << std::endl;
  for (size_t s=0; s<approxGradientWrtInput.n_slices; s++)
  {
    std::cout << DEBUG_PREFIX << "Slice #" << s << std::endl;
    for (size_t r=0; r<approxGradientWrtInput.slice(s).n_rows; r++)
      std::cout << DEBUG_PREFIX << approxGradientWrtInput.slice(s).row(r);
  }
#endif

  BOOST_REQUIRE(arma::approx_equal(gradInput,
                                   approxGradientWrtInput,
                                   "absdiff",
                                   disturbance));

  std::vector<arma::cube> approxGradientWrtFilters(2);
  approxGradientWrtFilters[0] = arma::zeros(3, 5, 3);
  approxGradientWrtFilters[1] = arma::zeros(3, 5, 3);

  std::vector<arma::cube> filters = c.getFilters();

  for (size_t fidx=0; fidx<2; fidx++)
  {
    for (size_t idx=0; idx<filters[fidx].n_elem; idx++)
    {
      filters[fidx][idx] += disturbance;
      c.setFilters(filters);
      c.Forward(input, output);
      double l1 = arma::accu(output);
      filters[fidx][idx] -= 2.0*disturbance;
      c.setFilters(filters);
      c.Forward(input, output);
      double l2 = arma::accu(output);
      approxGradientWrtFilters[fidx][idx] = (l1-l2)/(2.0*disturbance);
      filters[fidx][idx] += disturbance;
      c.setFilters(filters);
    }
  }

  for (size_t fidx=0; fidx<2; fidx++)
    BOOST_REQUIRE(arma::approx_equal(gradFilters[fidx],
                  approxGradientWrtFilters[fidx],
                  "absdiff",
                  disturbance));
}

#undef DEBUG
#undef DEBUG_PREFIX
