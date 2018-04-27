#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <cassert>

#define DEBUG false
#define DEBUG_PREFIX "[DEBUG DENSE LAYER ]\t"

class DenseLayer
{
 public:
  DenseLayer(size_t inputHeight,
             size_t inputWidth,
             size_t inputDepth,
             size_t numOutputs) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth),
      numOutputs(numOutputs)
  {
    // Initialize the weights.
    weights = arma::zeros(numOutputs, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );

    // Initialize the biases
    biases = arma::zeros(numOutputs);

    // Reset accumulated gradients.
    _resetAccumulatedGradients();
  }

  void Forward(arma::cube& input, arma::vec& output)
  {
    arma::vec flatInput = arma::vectorise(input);
    output = (weights * flatInput) + biases;

    this->input = input;
    this->output = output;
  }

  void Backward(arma::vec& upstreamGradient)
  {
    arma::vec gradInputVec = arma::zeros(inputHeight*inputWidth*inputDepth);
    for (size_t i=0; i<(inputHeight*inputWidth*inputDepth); i++)
      gradInputVec[i] = arma::dot(weights.col(i), upstreamGradient);
    arma::cube tmp((inputHeight*inputWidth*inputDepth), 1, 1);
    tmp.slice(0).col(0) = gradInputVec;
    gradInput = arma::reshape(tmp, inputHeight, inputWidth, inputDepth);

    accumulatedGradInput += gradInput;

    gradWeights = arma::zeros(arma::size(weights));
    for (size_t i=0; i<gradWeights.n_rows; i++)
      gradWeights.row(i) = vectorise(input).t() * upstreamGradient[i];

    accumulatedGradWeights += gradWeights;

    gradBiases = upstreamGradient;
    accumulatedGradBiases += gradBiases;
  }

  void UpdateWeightsAndBiases(size_t batchSize, double learningRate)
  {
    weights = weights - learningRate * (accumulatedGradWeights/batchSize);
    biases = biases - learningRate * (accumulatedGradBiases/batchSize);
    _resetAccumulatedGradients();
  }

  arma::mat getGradientWrtWeights() { return gradWeights; }

  arma::cube getGradientWrtInput() { return gradInput; }

  arma::vec getGradientWrtBiases() { return gradBiases; }

  arma::mat getWeights() { return weights; }

  arma::vec getBiases() { return biases; }

  void setWeights(arma::mat weights) { this->weights = weights; }

  void setBiases(arma::vec biases) { this->biases = biases; }

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  arma::cube input;

  size_t numOutputs;
  arma::vec output;

  arma::mat weights;
  arma::vec biases;

  arma::cube gradInput;
  arma::mat gradWeights;
  arma::vec gradBiases;

  arma::cube accumulatedGradInput;
  arma::mat accumulatedGradWeights;
  arma::vec accumulatedGradBiases;

  double _getTruncNormalVal(double mean, double variance)
  {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
      candidate.randn(1, 1);
    return candidate[0];
  }

  void _resetAccumulatedGradients()
  {
    accumulatedGradInput = arma::zeros(inputHeight, inputWidth, inputDepth);
    accumulatedGradWeights = arma::zeros(
        numOutputs,
        inputHeight*inputWidth*inputDepth
        );
    accumulatedGradBiases = arma::zeros(numOutputs);
  }
};

#undef DEBUG
#undef DEBUG_PREFIX

#endif
