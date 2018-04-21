#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG CONV LAYER ]\t"

class ConvolutionLayer
{
 public:
  ConvolutionLayer(
      size_t inputHeight,
      size_t inputWidth,
      size_t inputDepth,
      size_t filterHeight,
      size_t filterWidth,
      size_t horizontalStride,
      size_t verticalStride,
      size_t numFilters) :
    inputHeight(inputHeight),
    inputWidth(inputWidth),
    inputDepth(inputDepth),
    filterHeight(filterHeight),
    filterWidth(filterWidth),
    horizontalStride(horizontalStride),
    verticalStride(verticalStride),
    numFilters(numFilters)
  {
    // Initialize the filters.
    filters.resize(numFilters);
    for (size_t i=0; i<numFilters; i++)
    {
      filters[i] = arma::zeros(filterHeight, filterWidth, inputDepth);
      filters[i].imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );
    }
    _resetAccumulatedGradients();
#if DEBUG
    std::cout
        << DEBUG_PREFIX << "---------------------------------------------"
        << std::endl
        << DEBUG_PREFIX << "CONSTRUCTOR DEBUG OUTPUT"
        << std::endl
        << DEBUG_PREFIX << "---------------------------------------------"
        << std::endl;
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
#endif
  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    assert((inputHeight - filterHeight)%verticalStride == 0);
    assert((inputWidth - filterWidth)%horizontalStride == 0);
    output = arma::zeros((inputHeight - filterHeight)/verticalStride + 1,
                         (inputWidth - filterWidth)/horizontalStride + 1,
                         numFilters);
    for (size_t fidx = 0; fidx < numFilters; fidx++)
    {
      for (size_t i=0; i <= inputHeight - filterHeight; i += verticalStride)
        for (size_t j=0; j <= inputWidth - filterWidth; j += horizontalStride)
          output((i/verticalStride), (j/horizontalStride), fidx) = arma::dot(
              arma::vectorise(
                  input.subcube(i, j, 0,
                                i+filterHeight-1, j+filterWidth-1, inputDepth-1)
                ),
              arma::vectorise(filters[fidx]));
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

    // Print input.
    for (size_t i=0; i<inputDepth; i++)
    {
      std::cout << DEBUG_PREFIX << "Input slice #" << i << std::endl;
      for (size_t r=0; r<inputHeight; r++)
        std::cout << DEBUG_PREFIX << input.slice(i).row(r);
    }
    // Print filters.
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
    // Print output.
    for (size_t i=0; i<numFilters; i++)
    {
      std::cout << DEBUG_PREFIX << "Output slice #" << i << std::endl;
      for (size_t r=0; r<output.n_rows; r++)
        std::cout << DEBUG_PREFIX << output.slice(i).row(r);
    }
#endif
  }

  void Backward(arma::cube& upstreamGradient)
  {
    assert(upstreamGradient.n_slices == numFilters);
    assert(upstreamGradient.n_rows == output.n_rows);
    assert(upstreamGradient.n_cols == output.n_cols);

    gradInput = arma::zeros(arma::size(input));

    for (size_t sidx=0; sidx < numFilters; sidx++)
    {
      for (size_t r=0; r<output.n_rows; r ++)
      {
        for (size_t c=0; c<output.n_cols; c ++)
        {
          arma::cube tmp(arma::size(input), arma::fill::zeros);
          tmp.subcube(r*verticalStride, c*horizontalStride, 0, (r*verticalStride)+filterHeight-1, (c*horizontalStride)+filterWidth-1, inputDepth-1)
              = filters[sidx];
          gradInput += upstreamGradient.slice(sidx)(r, c) * tmp;
        }
      }
    }
#if DEBUG
    std::cout << DEBUG_PREFIX << "Gradient wrt input:" << std::endl;
    std::cout << gradInput << std::endl;
#endif

    accumulatedGradInput += gradInput;

    gradFilters.clear();
    gradFilters.resize(numFilters);
    for (size_t i=0; i<numFilters; i++)
      gradFilters[i] = arma::zeros(filterHeight, filterWidth, inputDepth);

    for (size_t fidx=0; fidx<numFilters; fidx++)
    {
      for (size_t r=0; r<output.n_rows; r ++)
      {
        for (size_t c=0; c<output.n_cols; c ++)
        {
          arma::cube tmp(arma::size(filters[fidx]), arma::fill::zeros);
          tmp = input.subcube(r*verticalStride, c*horizontalStride, 0,
              (r*verticalStride)+filterHeight-1, (c*horizontalStride)+filterWidth-1, inputDepth-1);
          gradFilters[fidx] += upstreamGradient.slice(fidx)(r, c) * tmp;
        }
      }
    }
#if DEBUG
    std::cout << DEBUG_PREFIX << "Gradient wrt filters:" << std::endl;
    for (size_t i=0; i<numFilters; i++)
      std::cout << gradFilters[i] << std::endl;
#endif

    for (size_t fidx=0; fidx<numFilters; fidx++)
      accumulatedGradFilters[fidx] += gradFilters[fidx];
  }

  void UpdateFilterWeights(size_t batchSize, double learningRate)
  {
    for (size_t fidx=0; fidx<numFilters; fidx++)
      filters[fidx] -= learningRate * (accumulatedGradFilters[fidx]/batchSize);

    _resetAccumulatedGradients();
  }

  void setFilters(std::vector<arma::cube> filters) { this->filters = filters; }

  std::vector<arma::cube> getFilters() { return this->filters; }

  arma::cube getGradientWrtInput() { return gradInput; }

  std::vector<arma::cube> getGradientWrtFilters() { return gradFilters; }

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  size_t filterHeight;
  size_t filterWidth;
  size_t horizontalStride;
  size_t verticalStride;
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

  void _resetAccumulatedGradients()
  {
    accumulatedGradFilters.clear();
    accumulatedGradFilters.resize(numFilters);
    for (size_t fidx=0; fidx<numFilters; fidx++)
      accumulatedGradFilters[fidx] = arma::zeros(filterHeight,
                                                 filterWidth,
                                                 inputDepth);
    accumulatedGradInput = arma::zeros(inputHeight, inputWidth, inputDepth);
  }

  bool _checkGradients()
  {
    // TODO
  }

  arma::cube input;
  arma::cube output;
  arma::cube gradInput;
  arma::cube accumulatedGradInput;
  std::vector<arma::cube> gradFilters;
  std::vector<arma::cube> accumulatedGradFilters;
};

#endif
