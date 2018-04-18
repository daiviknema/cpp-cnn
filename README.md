I'm at work and there's is nothing to do. So I'm going to write a CNN.. in C++ .. from scratch..

.. becuase that's what people do when they're bored out of their minds

So, my initial thoughts are:

1. I need some linear algebra library to make the vector stuff fast. I've worked with Eigen in the past, and mlpack uses Armadillo so I'm familiar with that too. I don't want to go through the hassle of installing Eigen (which may not be too much of a hassle, but still..) so I'm going to use Armadillo.

2. I'll need a class for the Convolution Layer, a class for the Pooling layer and a class for the dense layer at the end

3. Like all good boys who write CNNs, I think it would be best to test this on the MNIST dataset -- so we'll need code to parse that too. Since the data is stored in binary form, this might be unpleasant.. Oh well, I'll handle it when I get to it.

Let's begin right in the middle of everything - with the Convolution Layer.

So we've hit the ground running with the ConvolutionalLayer constructor -- It initializes stuff like dimensions of input volume, number of filters, stride and filter dimensions. It also initializes the filter weights. I've decided to use a truncated normal initialization (Ie. random values sampled from a Gaussian distribution having mean 0 and variance 1. Values more than two standard deviations away from the mean are rejected).

Cool cool .. the weight initializations look correct. Time for a git commit and then we move on to the forward pass through the conv layer.

I'm going to go for the simplest kind of convolution implementation there is -- no padding, no FFT.

I've realized that filters (in general) have a depth dimension as well, and I've defined filters as 2D matrices.. Need to fix that.

Done fixing the filter dimensions. Commit and move on to the forward pass.


