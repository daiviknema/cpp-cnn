#ifndef CNN_HPP
#define CNN_HPP

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG ]\t"

class ConvolutionLayer
{
 public:
  ConvolutionLayer(
      size_t inputHeight,
      size_t inputWidth,
      size_t inputDepth,
      size_t filterHeight,
      size_t filterWidth,
      size_t stride,
      size_t numFilters) :
    inputHeight(inputHeight),
    inputWidth(inputWidth),
    inputDepth(inputDepth),
    filterHeight(filterHeight),
    filterWidth(filterWidth),
    stride(stride),
    numFilters(numFilters)
  {
    // Initialize the filters.
    filters.resize(numFilters);
    for (size_t i=0; i<numFilters; i++)
    {
      filters[i] = arma::zeros(filterHeight, filterWidth, inputDepth);
      filters[i].imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );
    }
    if (DEBUG)
    {
      for (size_t i=0; i<numFilters; i++)
      {
        std::cout << DEBUG_PREFIX << "Filter #" << i << std::endl;
        std::cout << DEBUG_PREFIX << arma::size(filters[i]) << std::endl;
        for (size_t sidx=0; sidx<inputDepth; sidx++)
        {
          std::cout << DEBUG_PREFIX << "  Slice # " << sidx << std::endl;
          for (size_t ridx=0; ridx<filterHeight; ridx++)
            std::cout << DEBUG_PREFIX << filters[i].slice(sidx).row(ridx);
        }
      }
    }
  }

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  size_t filterHeight;
  size_t filterWidth;
  size_t stride;
  size_t numFilters;

  std::vector<arma::cube> filters;

  double _getTruncNormalVal(double mean, double variance)
  {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
      candidate.randn(1, 1);
    return candidate[0];
  }
};

class PoolingLayer
{
};

#endif
