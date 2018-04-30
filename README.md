## CPP-CNN

A C++ implementation of the popular LeNet convolutional neural network architecture. Currently it trains on the Kaggle Digit Recognizer challenge data and gives 0.973 accuracy on the leaderboard. At the time of writing this, I got a rank of 1414 using this model. The results csv file can be found in the `best-results/` directory.

I think that this is probably more for my own benefit than anyone else - but I've still tried to make to code as readable as possible in case someone else finds this and wants to play around with it.

### Prerequisites for building and running the model

You'll probably need
- g++ >= 5.0.0
- CMake >= 3.0.0
- make >= 4.0
- Armadillo >= 8.300.4
- Boost unit test framework (Boost version >= 1.58)

to run everything in this repo. I've only tried to run this on a Linux system (Ubuntu 16.04) -- but I dont see any obvious reason why it shouldn't work on other platforms as long as you have the dependencies installed.

You will also need the Kaggle Digit recognizer dataset - which can be downloaded from [here](https://www.kaggle.com/c/digit-recognizer/data)

### Building and Running the LeNet on the Digit Recognizer dataset

1. Clone this repository. `git clone https://github.com/plantsandbuildings/cpp-cnn`
2. `cd` into the project root (`cd cpp-cnn`) and create the build and data directories using `mkdir build data`.
3. Copy the Kaggle Digit Recognizer dataset into the `data` directory. The `data` directory should now contain two CSV files -- `train.csv` and `test.csv`.
4. `cd` into the build directory (`cd build`) and configure the build using `cmake ../` This will generate a `Makefile` to build the project.
5. Run `make` to build the project. Binaries are written to `build/bin`.
6. Train the model on the Kaggle data using `bin/le_net`.

The program will write the test predictions after each epoch of training into CSV files - `build/results_epoch_1.csv`, `build/results_epoch_2.csv` etc. These files can directly be uploaded to the [submission page](https://www.kaggle.com/c/digit-recognizer/submit) on Kaggle to view the scores.
