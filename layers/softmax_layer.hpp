#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include <iostream>
#include <armadillo>

class SoftmaxLayer
{
 public:
  SoftmaxLayer(size_t numInputs) :
      numInputs(numInputs)
  {
    // Nothing to do here.
  }

  void Forward(arma::vec& input, arma::vec& output)
  {
    double sumExp = arma::accu(arma::exp(input - arma::max(input)));
    output = arma::exp(input - arma::max(input))/sumExp;

    this->input = input;
    this->output = output;
  }

  void Backward(arma::vec& upstreamGradient)
  {
    double sub = arma::dot(upstreamGradient, output);
    gradWrtInput = (upstreamGradient - sub) % output;
  }

  arma::vec getGradientWrtInput() { return gradWrtInput; }

 private:
  size_t numInputs;
  arma::vec input;
  arma::vec output;

  arma::vec gradWrtInput;
};

#endif
