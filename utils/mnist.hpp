#ifndef MNIST_HPP
#define MNIST_HPP

#include <armadillo>
#include <string>
#include <vector>
#include <cassert>

class MNISTData
{
 public:
  MNISTData(std::string dataDir, double splitRatio = 0.9)
  {
    assert(splitRatio <= 1 && splitRatio >= 0);
    this->dataDir = dataDir;
    trainFile = dataDir + "/train.csv";
    testFile = dataDir + "/test.csv";

    arma::mat trainDataRaw;

    trainDataRaw.load(trainFile, arma::csv_ascii);
    trainDataRaw = trainDataRaw.submat(1, 0, trainDataRaw.n_rows - 1, trainDataRaw.n_cols - 1);

    int numExamples = trainDataRaw.n_rows;

    std::vector<arma::cube> trainDataAll;
    std::vector<arma::vec> trainLabelsAll;
    for (size_t idx=0; idx<trainDataRaw.n_rows; idx++)
    {
      int label = (int)(trainDataRaw.row(idx)(0));
      arma::cube img(28, 28, 1, arma::fill::zeros);
      for (size_t r=0; r<28; r++)
        img.slice(0).row(r) = trainDataRaw.row(idx).subvec(28*r+1, 28*r+28);
      img.slice(0) = arma::normalise(img.slice(0));
      trainDataAll.push_back(img);
      arma::vec labelvec(10, arma::fill::zeros);
      labelvec(label) += 1.0;
      trainLabelsAll.push_back(labelvec);
    }

    // Split trainDataAll and trainLabelsAll into train and validation parts.
    trainData = std::vector<arma::cube>(trainDataAll.begin(),
                                        trainDataAll.begin() + numExamples*splitRatio);
    trainLabels = std::vector<arma::vec>(trainLabelsAll.begin(),
                                         trainLabelsAll.begin() + numExamples*splitRatio);

    validationData = std::vector<arma::cube>(trainDataAll.begin() + numExamples*splitRatio,
                                             trainDataAll.end());
    validationLabels = std::vector<arma::vec>(trainLabelsAll.begin() + numExamples*splitRatio,
                                              trainLabelsAll.end());

    arma::mat testDataRaw;
    testDataRaw.load(testFile, arma::csv_ascii);
    testDataRaw = testDataRaw.submat(1, 0, testDataRaw.n_rows - 1, testDataRaw.n_cols - 1);
    for (size_t idx=0; idx<testDataRaw.n_rows; idx++)
    {
      arma::cube img(28, 28, 1, arma::fill::zeros);
      for (size_t r=0; r<28; r++)
        img.slice(0).row(r) = testDataRaw.row(idx).subvec(28*r, 28*r+27);
      img.slice(0) /= 255.0;
      testData.push_back(img);
    }
  }

  std::vector<arma::cube> getTrainData() { return trainData; }

  std::vector<arma::cube> getValidationData() { return validationData; }

  std::vector<arma::cube> getTestData() { return testData; }

  std::vector<arma::vec> getTrainLabels() { return trainLabels; }

  std::vector<arma::vec> getValidationLabels() { return validationLabels; }

 private:
  std::string dataDir;
  std::string trainFile;
  std::string testFile;

  std::vector<arma::cube> trainData;
  std::vector<arma::cube> validationData;
  std::vector<arma::cube> testData;

  std::vector<arma::vec> trainLabels;
  std::vector<arma::vec> validationLabels;
};

#endif
