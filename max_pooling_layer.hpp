#ifndef MAX_POOLING_LAYER_HPP
#define MAX_POOLING_LAYER_HPP

#include <iostream>
#include <armadillo>
#include <cassert>

#define DEBUG false
#define DEBUG_PREFIX "[DEBUG POOL LAYER ]\t"

class MaxPoolingLayer
{
 public:
  MaxPoolingLayer(size_t inputHeight,
                  size_t inputWidth,
                  size_t inputDepth,
                  size_t poolingWindowHeight,
                  size_t poolingWindowWidth,
                  size_t verticalStride,
                  size_t horizontalStride) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth),
      poolingWindowHeight(poolingWindowHeight),
      poolingWindowWidth(poolingWindowWidth),
      verticalStride(verticalStride),
      horizontalStride(horizontalStride)
  {
    // Nothing to do here.
  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    assert((inputHeight - poolingWindowHeight)%verticalStride == 0);
    assert((inputWidth - poolingWindowWidth)%horizontalStride == 0);
    output = arma::zeros(
        (inputHeight - poolingWindowHeight)/verticalStride + 1,
        (inputWidth - poolingWindowWidth)/horizontalStride + 1,
        inputDepth
        );
    for (size_t sidx = 0; sidx < inputDepth; sidx ++)
    {
      for (size_t ridx = 0;
           ridx <= inputHeight - poolingWindowHeight;
           ridx += verticalStride)
      {
        for (size_t cidx = 0;
             cidx <= inputWidth - poolingWindowWidth;
             cidx += horizontalStride)
        {
          output.slice(sidx)(ridx/verticalStride, cidx/horizontalStride) =
            input.slice(sidx).submat(ridx,
                          cidx,
                          ridx+poolingWindowHeight-1,
                          cidx+poolingWindowWidth-1)
            .max();
        }
      }
    }

    this->input = input;
    this->output = output;
#if DEBUG
    std::cout
        << DEBUG_PREFIX << "---------------------------------------------"
        << std::endl
        << DEBUG_PREFIX << "FORWARD PASS DEBUG OUTPUT"
        << std::endl
        << DEBUG_PREFIX << "---------------------------------------------"
        << std::endl;
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout
        << DEBUG_PREFIX << "Input to Max pooling layer:"
        << std::endl;
    for (size_t i=0; i<inputDepth; i++)
    {
      std::cout << DEBUG_PREFIX << "Slice #" << i << std::endl;
      for (size_t r=0; r < input.slice(i).n_rows; r++)
        std::cout << DEBUG_PREFIX << input.slice(i).row(r);
    }
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout
        << DEBUG_PREFIX << "Output of Max pooling layer:"
        << std::endl;
    for (size_t i=0; i<inputDepth; i++)
    {
      std::cout << DEBUG_PREFIX << "Slice #" << i << std::endl;
      for (size_t r=0; r < output.slice(i).n_rows; r++)
        std::cout << DEBUG_PREFIX << output.slice(i).row(r);
    }
#endif
  }

  void Backward(arma::cube& upstreamGradient)
  {
    assert (upstreamGradient.n_rows == output.n_rows);
    assert (upstreamGradient.n_cols == output.n_cols);
    assert (upstreamGradient.n_slices == output.n_slices);

    gradientWrtInput = arma::zeros(inputHeight, inputWidth, inputDepth);
    for (size_t i=0; i<inputDepth; i++)
    {
      for (size_t r=0;
           r + poolingWindowHeight <= inputHeight;
           r += verticalStride)
      {
        for (size_t c=0;
             c + poolingWindowWidth <= inputWidth;
             c += horizontalStride)
        {
          arma::mat tmp(poolingWindowHeight,
                         poolingWindowWidth,
                         arma::fill::zeros);
          tmp(input.slice(i).submat(r, c,
                r+poolingWindowHeight-1, c+poolingWindowWidth-1)
                .index_max()) = upstreamGradient.slice(i)(r/verticalStride,
                                                          c/horizontalStride);
          gradientWrtInput.slice(i).submat(r, c,
              r+poolingWindowHeight-1, c+poolingWindowWidth-1) += tmp;
        }
      }
    }
  }

  arma::cube getGradientWrtInput() { return gradientWrtInput; }

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  size_t poolingWindowHeight;
  size_t poolingWindowWidth;
  size_t verticalStride;
  size_t horizontalStride;

  arma::cube input;
  arma::cube output;

  arma::cube gradientWrtInput;
};

#undef DEBUG
#undef DEBUG_PREFIX
#endif
