#define BOOST_TEST_MODULE MNISTUtilTests
#define BOOST_TEST_DYN_LINK

#include "../utils/mnist.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
  MNISTData md("../data", 0.5);
}

