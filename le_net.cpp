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

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG LE NET ]\t"

int main(int argc, char ** argv)
{
  const double LEARNING_RATE = 0.0001;
  const size_t BATCH_SIZE = 1;

  MNISTData md("data_small", 0.9);

  std::vector<arma::cube> trainData = md.getTrainData();
  std::vector<arma::vec> trainLabels = md.getTrainLabels();

  std::vector<arma::cube> validationData = md.getValidationData();
  std::vector<arma::vec> validationLabels = md.getValidationLabels();

  std::vector<arma::cube> testData = md.getTestData();

  const size_t TRAIN_DATA_SZ = trainData.size();
  const size_t NUM_BATCHES = trainData.size() / BATCH_SIZE;

  std::cout << DEBUG_PREFIX << "Learning Rate: " << LEARNING_RATE << std::endl;
  std::cout << DEBUG_PREFIX << "Batch Size: " << BATCH_SIZE << std::endl;
  std::cout << DEBUG_PREFIX << "Training data size: " << TRAIN_DATA_SZ << std::endl;
  std::cout << DEBUG_PREFIX << "Number of batches: " << NUM_BATCHES << std::endl;

  // Define the network.
  ConvolutionLayer c1(
      28,
      28,
      1,
      5,
      5,
      1,
      1,
      6);
  // Output dims: 24 x 24 x 6
  ReLULayer r1(24, 24, 6);
  // Output dims: 24 x 24 x 6
  MaxPoolingLayer mp1(
      24,
      24,
      6,
      2,
      2,
      2,
      2);
  // Output dims: 12 x 12 x 6
  ConvolutionLayer c2(
      12,
      12,
      6,
      5,
      5,
      1,
      1,
      16);
  // Output dims: 8 x 8 x 16
  ReLULayer r2(8, 8, 16);
  // Output dims: 8 x 8 x 16
  MaxPoolingLayer mp2(
      8,
      8,
      16,
      2,
      2,
      2,
      2);
  // Output dims: 4 x 4 x 16
  DenseLayer d(
      4,
      4,
      16,
      10);
  // Output is a vector of size 10
  SoftmaxLayer s(10);
  // Output is a vector of size 10
  CrossEntropyLossLayer l(10);

  arma::cube c1Out;
  arma::cube r1Out;
  arma::cube mp1Out;
  arma::cube c2Out;
  arma::cube r2Out;
  arma::cube mp2Out;
  arma::vec dOut;
  arma::vec sOut;

  double loss;

  arma::vec gradWrtPredictedDistribution;
  arma::vec gradWrtSIn;
  arma::cube gradWrtDIn;
  arma::cube gradWrtMP2In;
  arma::cube gradWrtR2In;
  arma::cube gradWrtC2In;
  arma::cube gradWrtMP1In;
  arma::cube gradWrtR1In;
  arma::cube gradWrtC1In;

  for (size_t epoch=0; epoch<10; epoch++)
  {
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Epoch #" << epoch << std::endl;
    // Generate the batches randomly
    std::vector<arma::vec> batches(NUM_BATCHES);
    for (size_t i=0; i<NUM_BATCHES; i++)
    {
      batches[i] = arma::randu<arma::vec>(BATCH_SIZE);
      batches[i] *= (TRAIN_DATA_SZ - 1);
      batches[i] = arma::floor(batches[i]);
    }

    for (arma::vec batch: batches)
    {
      loss = 0.0;
      for (size_t i=0; i<batch.n_elem; i++)
      {
        // Forward pass the batch[i]th training example.
        c1.Forward(trainData[batch[i]], c1Out);
        r1.Forward(c1Out, r1Out);
        mp1.Forward(r1Out, mp1Out);
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        s.Forward(dOut, sOut);

        loss += l.Forward(sOut, trainLabels[batch[i]]);

        // Backward pass through the batch[i]th training example.
        l.Backward();
        gradWrtPredictedDistribution = l.getGradientWrtPredictedDistribution();
        s.Backward(gradWrtPredictedDistribution);
        gradWrtSIn = s.getGradientWrtInput();
        d.Backward(gradWrtSIn);
        gradWrtDIn = d.getGradientWrtInput();
        mp2.Backward(gradWrtDIn);
        gradWrtMP2In = mp2.getGradientWrtInput();
        r2.Backward(gradWrtMP2In);
        gradWrtR2In = r2.getGradientWrtInput();
        c2.Backward(gradWrtR2In);
        gradWrtC2In = c2.getGradientWrtInput();
        mp1.Backward(gradWrtC2In);
        gradWrtMP1In = mp1.getGradientWrtInput();
        r1.Backward(gradWrtMP1In);
        gradWrtR1In = r1.getGradientWrtInput();
        c1.Backward(gradWrtR1In);
        gradWrtC1In = c1.getGradientWrtInput();
      }

      // Update weights of c1, c2 and d
      c1.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      c2.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      d.UpdateWeights(BATCH_SIZE, LEARNING_RATE);
    }
    // Compute loss over training data
    loss = 0.0;
    for (size_t i=0; i<trainData.size(); i++)
    {
      c1.Forward(trainData[i], c1Out);
      r1.Forward(c1Out, r1Out);
      mp1.Forward(r1Out, mp1Out);
      c2.Forward(mp1Out, c2Out);
      r2.Forward(c2Out, r2Out);
      mp2.Forward(r2Out, mp2Out);
      d.Forward(mp2Out, dOut);
      s.Forward(dOut, sOut);

      loss += l.Forward(sOut, trainLabels[i]);
    }
    std::cout << DEBUG_PREFIX << "Training Data Loss: " << loss / trainData.size() << std::endl;
    // Compute loss over validation data
    loss = 0.0;
    for (size_t i=0; i<validationData.size(); i++)
    {
      c1.Forward(validationData[i], c1Out);
      r1.Forward(c1Out, r1Out);
      mp1.Forward(r1Out, mp1Out);
      c2.Forward(mp1Out, c2Out);
      r2.Forward(c2Out, r2Out);
      mp2.Forward(r2Out, mp2Out);
      d.Forward(mp2Out, dOut);
      s.Forward(dOut, sOut);

      loss += l.Forward(sOut, validationLabels[i]);
    }
    std::cout << DEBUG_PREFIX << "Validation Data Loss: " << loss / validationData.size() << std::endl;
    std::cout << DEBUG_PREFIX << std::endl;
  }

}

#undef DEBUG
#undef DEBUG_PREFIX
