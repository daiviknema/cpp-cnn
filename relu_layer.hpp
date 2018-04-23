#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include <iostream>
#include <armadillo>

class ReLULayer
{
 public:
  ReLULayer(size_t inputHeight,
       size_t inputWidth,
       size_t inputDepth) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth)
  {
    // Nothing to do here.
  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
    this->input = input;
    this->output = output;
  }

  void Backward(arma::cube upstreamGradient)
  {
    gradientWrtInput = input;
    gradientWrtInput.transform( [](double val) { return val > 0? 1 : 0; } );
    gradientWrtInput = gradientWrtInput % upstreamGradient;
  }

  arma::cube getGradientWrtInput() { return gradientWrtInput; }

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;

  arma::cube input;
  arma::cube output;

  arma::cube gradientWrtInput;
};

#endif
