#ifndef CROSS_ENTROPY_LOSS_LAYER_HPP
#define CROSS_ENTROPY_LOSS_LAYER_HPP

#include <iostream>
#include <cassert>
#include <armadillo>

class CrossEntropyLossLayer
{
 public:
  CrossEntropyLossLayer(size_t numInputs) : numInputs(numInputs)
  {
    // Nothing to do here.
  }

  double Forward(arma::vec& predictedDistribution,
                 arma::vec& actualDistribution)
  {
    assert(predictedDistribution.n_elem == numInputs);
    assert(actualDistribution.n_elem == numInputs);

    // Cache the prdicted and actual labels -- these will be required in the
    // backward pass.
    this->predictedDistribution = predictedDistribution;
    this->actualDistribution = actualDistribution;

    // Compute the loss and cache that too.
    this->loss = -arma::dot(actualDistribution,
                            arma::log(predictedDistribution));
    return this->loss;
  }

  void Backward()
  {
    gradientWrtPredictedDistribution =
        -(actualDistribution % (1/predictedDistribution));
  }

  arma::vec getGradientWrtPredictedDistribution()
  {
    return gradientWrtPredictedDistribution;
  }

 private:
  size_t numInputs;
  arma::vec predictedDistribution;
  arma::vec actualDistribution;

  double loss;

  arma::vec gradientWrtPredictedDistribution;
};

#endif
