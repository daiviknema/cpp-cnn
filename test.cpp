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
