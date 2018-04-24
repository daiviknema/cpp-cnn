#define BOOST_TEST_MODULE IntegrationTests
#define BOOST_TEST_DYN_LINK

#include "conv_layer.hpp"
#include "max_pooling_layer.hpp"
#include "relu_layer.hpp"
#include "dense_layer.hpp"
#include "softmax_layer.hpp"
#include "cross_entropy_loss_layer.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG INTEGRATION TEST ]\t"

BOOST_AUTO_TEST_CASE(SimpleNetworkTest)
{
  // Generate some dummy training data.
  std::vector<arma::cube> trainData;

  arma::cube trainExample1(5, 7, 1);
  arma::mat pos(5, 7, arma::fill::zeros);
  pos.col(1) = arma::ones(5);
  trainExample1.slice(0) = pos;
  trainData.push_back(trainExample1);

  arma::cube trainExample2(5, 7, 1);
  arma::mat neg(5, 7, arma::fill::randn);
  neg = arma::normalise(neg);
  trainExample2.slice(0) = neg;
  trainData.push_back(trainExample2);

  std::vector<arma::vec> trainLabels;

  arma::vec pos_ = {1, 0};
  arma::vec neg_ = {0, 1};
  trainLabels.push_back(pos_);
  trainLabels.push_back(neg_);

  // Define the network.
  ConvolutionLayer c(
      5,
      7,
      1,
      3,
      2,
      1,
      2,
      4);
  // Output dims: 2 x 6 x 4
  ReLULayer r(2, 6, 4);
  // Output dims: 2 x 6 x 4
  DenseLayer d(
      2,
      6,
      4,
      2);
  // Output is a vector of size 2
  SoftmaxLayer s(2);
  // Output is a vector of size 2
  CrossEntropyLossLayer l(2);

  arma::cube convOut;
  arma::cube reluOut;
  arma::vec denseOut;
  arma::vec softmaxOut;
  double loss;


  arma::vec gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
  arma::vec gradWrtSoftmaxInput;
  arma::cube gradWrtDenseInput;
  arma::cube gradWrtReluInput;
  arma::cube gradWrtConvInput;
  for (size_t epoch=0; epoch<10; epoch++)
  {
    // Forward pass the first example.
    c.Forward(trainData[0], convOut);
    r.Forward(convOut, reluOut);
    d.Forward(reluOut, denseOut);
    s.Forward(denseOut, softmaxOut);
    loss = l.Forward(softmaxOut, trainLabels[0]);

    // Backward pass through the first example.
    l.Backward();
    gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
    s.Backward(gradWrtPredictedDistribution);
    gradWrtSoftmaxInput = s.getGradientWrtInput();
    d.Backward(gradWrtSoftmaxInput);
    gradWrtDenseInput = d.getGradientWrtInput();
    r.Backward(gradWrtDenseInput);
    gradWrtReluInput = r.getGradientWrtInput();
    c.Backward(gradWrtReluInput);
    gradWrtConvInput = c.getGradientWrtInput();

    // Forward pass the second example.
    c.Forward(trainData[1], convOut);
    r.Forward(convOut, reluOut);
    d.Forward(reluOut, denseOut);
    s.Forward(denseOut, softmaxOut);
    loss = l.Forward(softmaxOut, trainLabels[1]);

    // Backward pass through the second example.
    l.Backward();
    gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
    s.Backward(gradWrtPredictedDistribution);
    gradWrtSoftmaxInput = s.getGradientWrtInput();
    d.Backward(gradWrtSoftmaxInput);
    gradWrtDenseInput = d.getGradientWrtInput();
    r.Backward(gradWrtDenseInput);
    gradWrtReluInput = r.getGradientWrtInput();
    c.Backward(gradWrtReluInput);
    gradWrtConvInput = c.getGradientWrtInput();

    // Update weights.
    d.UpdateWeights(2, 0.1);
    c.UpdateFilterWeights(2, 0.1);

#if DEBUG
    std::cout << DEBUG_PREFIX << "Epoch #" << epoch
        << "\tCross Entropy Loss: " << loss << std::endl;
#endif
  }
}

#undef DEBUG
#undef DEBUG_PREFIX
