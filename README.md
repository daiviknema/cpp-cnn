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

Update: I went for a snack break. My boss told me to do a few things -- which I mostly ignored, because he's an idiot. And I just finished the forward pass implementation. It feels weird working on this in the office, so I'm going to head home and continue from there. I plan to test the forward pass implementation first and then try and figure out how the backward pass is going to go.

A few more notes on the rest of the implementation:
1. The optimizer -- for now, I'll just use vanilla mini-batch SGD  to train. Maybe later I'll switch it up to Adam or RMSprop.

2. I realize that I'll need to add layers for ReLU activation in the hidden units, and a softmax layer at the very end of the network.

3. I'll probably add a class like "class LeNet" that contains the entire CNN architecture. I dont really plan on reusing any of the layers, so its fine if they're a bit dirty.

3 minutes till my cab arrives .. better head down.

Okay, I didn't really do much at home yesterday, and today I was a bit busy doing pointless things at work. I found some time now to work on this, and have completed the backward pass. Actually, I'd been thinking about the math of the backward pass through the conv layer today.. and I was quite surprised at how easily it worked out to nice expressions. Hopefully I'll get some time to write a blog post about it -- I think I actually found a nice method to it as well. Anyway, I'd scribbled down
most of it in my little notebook at work and the implementation wasn't too hard.

I've still got the forward pass testing in the backlog. Now, I can add the backward pass testing as well. I've added a function stub for gradient check as well in the CovolutionLayer class. I think I'll make 100% sure that theres nothing wring with my Conv layer before proceeding with the other components (dense layer, max pooling, relu and sigmoid). Hopefully tomorrow I'll get time for testing and then finish the CNN over the weekend.

Oh yeah, a minor note -- I'd not differentiated between the strides in the vertical and horizontal directions. Updated this.

I've been giving a bit of thought to writing proper tests -- test driven development and all that. I've decided to go with the boost.Test framework (which also happens to be used by mlpack).

It WORKS!!! Both the forward and backward pass seem to be working fine on basic tests. I've even written a gradient check and both the analytic and numeric gradients agree. I didn't expect things to go so smoothly, I was completely prepared to shed tears -- but hey, looks like I'm smart after all.

I've added a larger test for the backward pass - and used different prime values for input dimensions and filter dimensions. This was a good test to add.. it pointed out a bug in the way gradients were being propagated when the stride was > 1.

With this, the convolution layer looks pretty much done to me - atleast for now. I can proceed with the dense layer now. I think I'll refactor the project into more files.

Oh wait, there need to be more tests for checking if the gradients are being accumulated and if the batch update is happening correctly. I'll add those after some part of the dense layer implementation.

Yeah.. I went to sleep after I got the conv layer working. Today is saturday, so I've been at this for a while -- and here's the update. I've completed the dense layer implementation and written tests for all of it. It works really nicely. I've also been giving some thought to how I'm going to parse the MNIST data -- I think I'll just use mlpack's data::Load functionality to load the data into armadillo matrices and then let my CNN model take it from there. I really dont want to be writing
code to parse a binary file into an armadillo matrix by myself. Also, I think in the gradient check I did for the conv layer backward pass, I checked only the gradient wrt input -- and completely forgot about the gradient wrt filters. I'm going to add that now.

Next on the TODO list would be the implementation of softmax layer, max pooling layer and relu layer (in that order).

Added gradient wrt filters check in the conv layer backward pass test. As expected, gradients are correct. I've also remembered that I need to add the update weights function in the dense layer, and write tests to check the updates.

Added softmax layer and tests for forward and backward passes.

Okay, so yesterday was SUnday and I didn't do shit. I was at work all day today and after that I was waiting for the Google Summer of Code results to be announced (too excited to get any work done). Anyway, the results were a big let down - not that I was expecting a selection.. still, rejection hurts. I'm back at it with the MaxPooling layer implementation. The backward pass proved to be trickier than expected but I think I've got it right -- I'm going to write the tests for it now, and then
we'll know for sure.

Yep. It works. I guess only RelU remains now.

ReLU Done! All the components are done. I think I'll write some more tests to try out small networks before I create LeNet.

Actually, one component is still remains -- the loss layer. Imma have to implement cross entropy loss first

Implemented cross entropy loss along with tests. Now we can move to the integration tests.

I went to sleep last night.. I've decided to skip office today because I'm still a little bummed at the GSoC rejection. Anyway, the silver lining is that all my components seem to be working - I just wrote a simple network and ran it on two training examples for 10 epochs and the loss decreases beautifully:
```
[DEBUG INTEGRATION TEST ]       Epoch #0        Cross Entropy Loss: 0.482423
[DEBUG INTEGRATION TEST ]       Epoch #1        Cross Entropy Loss: 0.121352
[DEBUG INTEGRATION TEST ]       Epoch #2        Cross Entropy Loss: 0.0802145
[DEBUG INTEGRATION TEST ]       Epoch #3        Cross Entropy Loss: 0.0604326
[DEBUG INTEGRATION TEST ]       Epoch #4        Cross Entropy Loss: 0.0488966
[DEBUG INTEGRATION TEST ]       Epoch #5        Cross Entropy Loss: 0.0410873
[DEBUG INTEGRATION TEST ]       Epoch #6        Cross Entropy Loss: 0.0354383
[DEBUG INTEGRATION TEST ]       Epoch #7        Cross Entropy Loss: 0.0311572
[DEBUG INTEGRATION TEST ]       Epoch #8        Cross Entropy Loss: 0.0277985
[DEBUG INTEGRATION TEST ]       Epoch #9        Cross Entropy Loss: 0.0250919
```

I think I should document the code and make style fixes before I proceed with anything else.

Okay, I've refactored the code and made a bunch of style fixes and added comments (sparingly). Also, I've added a cmake configuration to easily build everything and make things cross-platform. Now for the part that I've been putting off from the very beginning: parsing the binary MNIST data.

I've just realized that the digit recognizer challenge on Kaggle has csv datasets for digit recognition. Those might be easier to parse.

Done with the data parsing module. Now for the big sausage - LeNet. Oh yeah, minor note -- anyone attempting to run this code will have to download the Kaggle dataset into a `data/` directory.

Okay, so I've assembled the Le Net - but there seems to be a very strange issue.. The training loss decreses over epochs, so does the validation loss - all good, right? Wrong! The training and validation accuracies are also decresing over epochs! FTW!!! Go home CNN, you're drunk! What is strange is that I can't seem to get the model to overfit on a smaller sub-dataset either. I think its time to write another integration test.

I might've made some headway into the issue - it looks like the input to the loss layer is very very close to a one-hot vector which is causing infinities and negative infinities to appear. Need to find some way to make this numerically stable. Okay, a little googling around has shown that if we combine the softmax and cross entropy layers then the backward gradient becomes numerically stable. So, we will do that now.

It's not working at all. Need to start fresh.

Okay, so I've written a few more integration tests and here is what I've found:
- The backward pass through the dense layer was slightly incorrect. I'd forgotten to incorporate the upstreamGradient into the gradients wrt weights.
- The dense layer was also missing biases. I've added these now.
With these changes, I can train simple networks for:
- learning the AND decision boundary
- learning a single MNIST image
- On a sample of 3000 MNIST images, a simple convnet (conv-relu-maxpool-dense-softmax-cross_entropy schema) can be trained with SGD to obtain the following results:
```
[DEBUG INTEGRATION TEST ]	Size of training set: 2700
[DEBUG INTEGRATION TEST ]	Size of validation set: 300
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 2.22893
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.406667
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 1.33203
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.676667
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.841367
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.753333
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.584995
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.79
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.44068
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.813333
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.360519
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.81
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.294253
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.84
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.265645
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.83
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.220504
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.863333
[DEBUG INTEGRATION TEST ]
[DEBUG INTEGRATION TEST ]	Average loss: 0.164675
[DEBUG INTEGRATION TEST ]	Validation Accuracy: 0.863333
```

which is reassuring ... I think we should be good to go on LeNet now.

Fuck yeah .. LeNet is working on the medium sozed dataset:
```
[DEBUG LE NET ] Training data size: 2700
[DEBUG LE NET ] Validation data size: 300
[DEBUG LE NET ] Test data size: 10
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #0: 0.578797
[DEBUG LE NET ] Val accuracy: 0.886667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #1: 0.181674
[DEBUG LE NET ] Val accuracy: 0.936667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #2: 0.155978
[DEBUG LE NET ] Val accuracy: 0.913333
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #3: 0.0978818
[DEBUG LE NET ] Val accuracy: 0.956667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #4: 0.0800541
[DEBUG LE NET ] Val accuracy: 0.953333
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #5: 0.0567186
[DEBUG LE NET ] Val accuracy: 0.936667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #6: 0.0514032
[DEBUG LE NET ] Val accuracy: 0.916667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #7: 0.0396252
[DEBUG LE NET ] Val accuracy: 0.926667
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #8: 0.0444968
[DEBUG LE NET ] Val accuracy: 0.933333
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #9: 0.0350243
[DEBUG LE NET ] Val accuracy: 0.93
```
For future reference -- these results were obtained with the following hyperparameter settings:
- Learning rate: 0.05
- Epochs: 10
- Batch Size: 10
- Train data: data_medium
Oh, and there's one more thing.. I noticed earlier that the output of the dense layer is quite high -- of the order of 1e2. Clearly too high for the softmax to give meaningful outputs. So, I've scaled the input to the softmax by 1e2. This is hacky, and I should probably figure out a cleaner way to do this. Maybe normalize the input differently?.. I think I might try to make the input have zero mean and unit variance and try - but for now I think the scaling is fine.

Now for the mother lode... the complete Kaggle dataset

OMFG!!!! IT WOOORRKKSS!!!!
It's completed 5 epochs:
```
[DEBUG LE NET ] Training data size: 37800
[DEBUG LE NET ] Validation data size: 4200
[DEBUG LE NET ] Test data size: 28000
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #0: 0.189032
[DEBUG LE NET ] Val accuracy: 0.960952
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #1: 0.102551
[DEBUG LE NET ] Val accuracy: 0.966905
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #2: 0.0846397
[DEBUG LE NET ] Val accuracy: 0.971905
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #3: 0.0762915
[DEBUG LE NET ] Val accuracy: 0.97119
[DEBUG LE NET ]
[DEBUG LE NET ] Loss after epoch #4: 0.0741992
[DEBUG LE NET ] Val accuracy: 0.975714
```
I think that this is not bad at all for a handwritten CNN. It takes a long time to run (~20 minutes per epoch on my shitty machine), but then again - handwritten. I'm going to stop the execution now and save the results and make the debug output prettier. For future reference: I did not change the hyperparameters from the previous run -- only the dataset was expanded to the original size.
