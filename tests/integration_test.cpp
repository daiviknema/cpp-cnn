#define BOOST_TEST_MODULE IntegrationTests
#define BOOST_TEST_DYN_LINK

#include "../layers/convolution_layer.hpp"
#include "../layers/max_pooling_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/cross_entropy_loss_layer.hpp"

#include "../utils/mnist.hpp"

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


  arma::vec gradWrtPredictedDistribution =
      l.getGradientWrtPredictedDistribution();
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
    loss += l.Forward(softmaxOut, trainLabels[0]);

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
    loss += l.Forward(softmaxOut, trainLabels[1]);

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
    d.UpdateWeightsAndBiases(2, 0.1);
    c.UpdateFilterWeights(2, 0.1);

#if DEBUG
    std::cout << DEBUG_PREFIX << "Epoch #" << epoch
        << "\tCross Entropy Loss: " << loss << std::endl;
#endif
    loss = 0.0;
  }
#if DEBUG
  // Let us have a look at the peridctions
  c.Forward(trainData[0], convOut);
  r.Forward(convOut, reluOut);
  d.Forward(reluOut, denseOut);
  s.Forward(denseOut, softmaxOut);
  std::cout << DEBUG_PREFIX << softmaxOut.t();
  c.Forward(trainData[1], convOut);
  r.Forward(convOut, reluOut);
  d.Forward(reluOut, denseOut);
  s.Forward(denseOut, softmaxOut);
  std::cout << DEBUG_PREFIX << softmaxOut.t();
#endif

}

BOOST_AUTO_TEST_CASE(SmallANDNetwork)
{
  std::vector<arma::cube> trainData(4, arma::cube(2, 1, 1, arma::fill::zeros));
  trainData[1].slice(0).col(0) = arma::vec({1, 0});
  trainData[2].slice(0).col(0) = arma::vec({0, 1});
  trainData[3].slice(0).col(0) = arma::vec({1, 1});

  std::vector<arma::vec> trainLabels(4);
  trainLabels[0] = {1, 0};
  trainLabels[1] = {1, 0};
  trainLabels[2] = {1, 0};
  trainLabels[3] = {0, 1};

  DenseLayer d(2, 1, 1, 2);
  SoftmaxLayer s(2);
  CrossEntropyLossLayer l(2);

  arma::vec dOut = arma::zeros(2);
  arma::vec sOut = arma::zeros(2);
  double loss = 0.0;

  for (size_t epoch = 0; epoch < 1000; epoch ++)
  {
    loss = 0.0;
    for (size_t i=0; i<4; i++)
    {
      d.Forward(trainData[i], dOut);
      s.Forward(dOut, sOut);
      loss += l.Forward(sOut, trainLabels[i]);

      std::cout << DEBUG_PREFIX << std::endl;
      std::cout << DEBUG_PREFIX << "Input: " << trainData[i].slice(0).col(0).t();
      std::cout << DEBUG_PREFIX << "Target: " << trainLabels[i].t();
      std::cout << DEBUG_PREFIX << "Predicted: " << sOut.t();

      l.Backward();
      arma::vec gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
      s.Backward(gradWrtPredictedDistribution);
      arma::vec gradWrtSIn = s.getGradientWrtInput();
      d.Backward(gradWrtSIn);
      arma::vec gradWrtDin = d.getGradientWrtInput();
      arma::mat gradWrtWeights = d.getGradientWrtWeights();

      std::cout << DEBUG_PREFIX << "Gradient wrt weights:" << std::endl;
      std::cout << gradWrtWeights << std::endl;
    }
    std::cout << DEBUG_PREFIX << "Weights before update:" << std::endl;
    std::cout << d.getWeights() << std::endl;
    std::cout << DEBUG_PREFIX << "Biases before update:" << std::endl;
    std::cout << d.getBiases() << std::endl;
    d.UpdateWeightsAndBiases(4, 0.1);
    std::cout << DEBUG_PREFIX << "Weights after update:" << std::endl;
    std::cout << d.getWeights() << std::endl;
    std::cout << DEBUG_PREFIX << "Biases after update:" << std::endl;
    std::cout << d.getBiases() << std::endl;
    std::cout << DEBUG_PREFIX << "Loss after epoch #" << epoch << ": " << loss << std::endl;
  }
  // Now we check the predictions
  for (size_t i=0; i<4; i++)
  {
    d.Forward(trainData[i], dOut);
    s.Forward(dOut, sOut);

    std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Input: " << arma::vectorise(trainData[i]).t();
    std::cout << DEBUG_PREFIX << "Prediction: " << sOut.t();
    std::cout << DEBUG_PREFIX << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(MNISTSmallDenseNetworkTest)
{
  MNISTData md("../data_small");

  std::vector<arma::cube> trainData = md.getTrainData();
  std::vector<arma::vec> trainLabels = md.getTrainLabels();

  std::vector<arma::cube> validationData = md.getValidationData();
  std::vector<arma::vec> validationLabels = md.getValidationLabels();

  const size_t TRAINING_DATA_SIZE = trainData.size();
  const size_t VALIDATION_DATA_SIZE = validationData.size();

  std::cout << "Training Data size: " << TRAINING_DATA_SIZE << std::endl;
  std::cout << "Validation Data size: " << VALIDATION_DATA_SIZE << std::endl;

  DenseLayer d(28, 28, 1, 10);
  SoftmaxLayer s(10);
  CrossEntropyLossLayer l(10);

  arma::vec dOut = arma::zeros(10);
  arma::vec sOut = arma::zeros(10);

  arma::mat oldWts = arma::zeros(10, 28*28*1);
  arma::mat newWts = arma::zeros(10, 28*28*1);

  arma::vec oldDOut = arma::zeros(10);
  arma::vec newDOut = arma::zeros(10);

  arma::vec oldSOut = arma::zeros(10);
  arma::vec newSOut = arma::zeros(10);

  // Forward pass the first training example.
  for (size_t epoch = 0; epoch < 100; epoch++)
  {
    oldDOut = dOut;
    d.Forward(trainData[0], dOut);
    newDOut = dOut;
    BOOST_REQUIRE(!arma::approx_equal(oldDOut, newDOut, "absdiff", 0.0));

    oldSOut = sOut;
    s.Forward(dOut, sOut);
    newSOut = sOut;
    BOOST_REQUIRE(!arma::approx_equal(oldSOut, newSOut, "absdiff", 0.0));
    std::cout << DEBUG_PREFIX << "Old softmax output:" << std::endl;
    std::cout << oldSOut << std::endl;
    std::cout << DEBUG_PREFIX << "New softmax output:" << std::endl;
    std::cout << newSOut << std::endl;

    double loss = l.Forward(sOut, trainLabels[0]);

    // std::cout << DEBUG_PREFIX << "Input to dense layer:" << std::endl;
    // std::cout << trainData[0] << std::endl;

    // std::cout << DEBUG_PREFIX << "Weights of dense layer:" << std::endl;
    // std::cout << d.getWeights() << std::endl;

    // std::cout << DEBUG_PREFIX << "Output of dense layer:" << std::endl;
    // std::cout << sOut << std::endl;

    std::cout << DEBUG_PREFIX << "Loss: " << loss << std::endl;

    l.Backward();
    arma::vec gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();

    // std::cout << DEBUG_PREFIX << "Gradient wrt predicted distribution:" << std::endl;
    // std::cout << gradWrtPredictedDistribution << std::endl;

    s.Backward(gradWrtPredictedDistribution);
    arma::vec gradWrtSIn = s.getGradientWrtInput();

    // std::cout << DEBUG_PREFIX << "Gradient wrt softmax input:"  << std::endl;
    // std::cout << gradWrtSIn << std::endl;

    d.Backward(gradWrtSIn);
    arma::mat gradWrtWts = d.getGradientWrtWeights();

    // std::cout << DEBUG_PREFIX << "Gradient wrt dense weights:" << std::endl;
    // std::cout << gradWrtWts << std::endl;

    oldWts = d.getWeights();
    d.UpdateWeightsAndBiases(1, 0.1);
    newWts = d.getWeights();
    BOOST_REQUIRE(!arma::approx_equal(oldWts, newWts, "absdiff", 0.0));
  }

  std::cout << DEBUG_PREFIX << std::endl;
  d.Forward(trainData[0], dOut);
  s.Forward(dOut, sOut);
  std::cout << DEBUG_PREFIX << "Actual output: " << trainLabels[0].t();
  std::cout << DEBUG_PREFIX << "Predicted output: " << sOut.t();
}

BOOST_AUTO_TEST_CASE(NowWereGettingSomewhereTest)
{
  MNISTData md("../data_medium");

  std::vector<arma::cube> trainData = md.getTrainData();
  std::vector<arma::vec> trainLabels = md.getTrainLabels();

  std::vector<arma::cube> validationData = md.getValidationData();
  std::vector<arma::vec> validationLabels = md.getValidationLabels();

  std::cout << DEBUG_PREFIX << "Size of training set: " << trainData.size() << std::endl;
  BOOST_REQUIRE_EQUAL(trainData.size(), trainLabels.size());
  std::cout << DEBUG_PREFIX << "Size of validation set: " << validationData.size() << std::endl;
  BOOST_REQUIRE_EQUAL(validationData.size(), validationLabels.size());

  // Define the network
  // conv - relu - maxpool - dense - softmax - loss

  ConvolutionLayer c(
      28,
      28,
      1,
      7,
      7,
      1,
      1,
      3);
  // Output is 22 x 22 x 3
  ReLULayer r(
      22,
      22,
      3);
  // Output is 22 x 22 x 3
  MaxPoolingLayer m(
      22,
      22,
      3,
      2,
      2,
      2,
      2);
  // Output is 11 x 11 x 3
  DenseLayer d(
      11,
      11,
      3,
      10);
  // Output is a vector of size 10
  SoftmaxLayer s(10);
  // Output is a vector of size 10
  CrossEntropyLossLayer l(10);

  arma::cube cOut = arma::zeros(22, 22, 3);
  arma::cube rOut = arma::zeros(22, 22, 3);
  arma::cube mOut = arma::zeros(11, 11, 3);
  arma::vec dOut = arma::zeros(10);
  arma::vec sOut = arma::zeros(10);
  double loss = 0.0;
  // We'll use stochastic gradient descent
  for (size_t epoch = 0; epoch < 10; epoch++)
  {
    double averageLoss = 0.0;
    for(size_t i=0; i<trainData.size(); i++)
    {
      c.Forward(trainData[i], cOut);
      r.Forward(cOut, rOut);
      m.Forward(rOut, mOut);
      d.Forward(mOut, dOut);
      s.Forward(dOut, sOut);

      loss = l.Forward(sOut, trainLabels[i]);
      averageLoss += loss;

      l.Backward();
      arma::vec gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
      s.Backward(gradWrtPredictedDistribution);
      arma::vec gradWrtSIn = s.getGradientWrtInput();
      d.Backward(gradWrtSIn);
      arma::cube gradWrtDIn = d.getGradientWrtInput();
      m.Backward(gradWrtDIn);
      arma::cube gradWrtMIn = m.getGradientWrtInput();
      r.Backward(gradWrtMIn);
      arma::cube gradWrtRIn = r.getGradientWrtInput();
      c.Backward(gradWrtRIn);
      arma::cube gradWrtCIn = c.getGradientWrtInput();

      d.UpdateWeightsAndBiases(1, 0.1);
      c.UpdateFilterWeights(1, 0.1);
    }
    averageLoss /= trainData.size();
    std::cout << DEBUG_PREFIX << "Average loss: " << averageLoss << std::endl;
    // Compute the validation accuracy
    double correct = 0.0;
    for (size_t i=0; i<validationData.size(); i++)
    {
      c.Forward(validationData[i], cOut);
      r.Forward(cOut, rOut);
      m.Forward(rOut, mOut);
      d.Forward(mOut, dOut);
      s.Forward(dOut, sOut);

      if (sOut.index_max() == validationLabels[i].index_max())
        correct += 1.0;
    }
    std::cout << DEBUG_PREFIX << "Validation Accuracy: " << correct/validationData.size() << std::endl;
    std::cout << DEBUG_PREFIX << std::endl;
  }
}

#undef DEBUG
#undef DEBUG_PREFIX
