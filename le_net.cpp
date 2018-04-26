#include "layers/convolution_layer.hpp"
#include "layers/max_pooling_layer.hpp"
#include "layers/relu_layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/cross_entropy_loss_layer.hpp"
#include "utils/mnist.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <fstream>

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG LE NET ]\t"

int main(int argc, char ** argv)
{
}

#undef DEBUG
#undef DEBUG_PREFIX
