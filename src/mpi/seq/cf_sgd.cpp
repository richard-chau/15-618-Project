#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include "cf_sgd.h"
#include "CycleTimer.h"



class DataReader100K {
 public:
  DataReader100K(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> buffer;
    int userId, itemId, rating, timestamp;
    while (file >> userId >> itemId >> rating >> timestamp) {
      // In file, userId and itemId start from 1.
      buffer.emplace_back(userId-1);
      buffer.emplace_back(itemId-1);
      buffer.emplace_back(rating);
    }
    
    int nTrain = buffer.size() / 3;
    DEBUG_ASSERT(nTrain * 3 == buffer.size());
    ratings.resize(3, nTrain);
    for (int c = 0; c < nTrain; c++) {
      for (int r = 0; r < 3; r++) {
        ratings(r, c) = buffer[3*c+r];
      }
    }
  }

  DataType&& MoveData() {
    return std::move(ratings);
  }

 private:
  // Each column is a data sample:
  // ---
  // user id
  // ---
  // item id
  // ---
  // rating
  // ---
  DataType ratings;
};

class DataReader10M {
 public:
  DataReader10M(const std::string& filename) {
    std::ifstream file(filename);
    int userId, itemId;
    double rating;
    unsigned long long timestamp;
    std::vector<double> buffer;
    
    FILE* fp = fopen(filename.c_str(), "r");
    while (fscanf(fp, "%d::%d::%lf::%llu",
        &userId, &itemId, &rating, &timestamp) == 4) {
      // In file, userId and itemId start from 1.
      buffer.emplace_back(userId-1);
      buffer.emplace_back(itemId-1);
      buffer.emplace_back(rating);
    }
    fclose(fp);

    int nTrain = buffer.size() / 3;
    DEBUG_ASSERT(nTrain * 3 == buffer.size());
    ratings.resize(3, nTrain);
    for (int c = 0; c < nTrain; c++) {
      for (int r = 0; r < 3; r++) {
        ratings(r, c) = buffer[3*c+r];
      }
    }
  }

  DataType&& MoveData() {
    return std::move(ratings);
  }

  const DataType& Data() {
    return ratings;
  }

 private:
  // Each column is a data sample:
  // ---
  // user id
  // ---
  // item id
  // ---
  // rating
  // ---
  DataType ratings;
};


class SGD {
 public:
  SGD(DataType&& trainData_, double lambda_, double stepSize_, int nEpoch_):
      ratings(std::move(trainData_)), lambda(lambda_),
      stepSize(stepSize_), nEpoch(nEpoch_) { }
  
  void Train() {
    InitializeParam();
    double l = std::sqrt(ComputeLoss());
    LOG("Initial RMSE = %lf\n", l);
    for (int epoch = 0; epoch < nEpoch; epoch++) {
      // LOG("Start epoch %d\n", epoch);
      double t1 = CycleTimer::currentSeconds();
      Iterate();
      double t2 = CycleTimer::currentSeconds();
      double loss = ComputeLoss();
      double t3 = CycleTimer::currentSeconds();
      loss = std::sqrt(loss);
      losses.emplace_back(loss);

      LOG("Epoch %d loss=%lf, iteration time=%lf, loss computation time=%lf\n",
          epoch, loss, t2-t1, t3-t2);
    }
  }

  double TestRMSE(const DataType& testData) {
    int nTest = testData.cols();

    double squaredError = 0.0;
    for (int i = 0; i < nTest; i++) {
      int user = static_cast<int>(ratings(0, i));
      int item = static_cast<int>(ratings(1, i));
      double rating = ratings(2, i);

      double diff = rating - userParam.col(user).dot(itemParam.col(item));
      squaredError += diff * diff;
    }

    double rmse = std::sqrt(squaredError / nTest);
    return rmse;
  }

 private:
  DataType ratings;
  ParamType userParam;
  ParamType itemParam;
  int nUser;
  int nItem;
  std::vector<double> losses;
  std::vector<int> perm;
  double lambda; // regularization term
  double stepSize;
  int nEpoch;

  std::vector<int> numUserRating;
  std::vector<int> numItemRating;

  void InitializeParam() {
    // Initialize parameter.
    const int nTrain = ratings.cols();
    Eigen::VectorXd maxVal = ratings.rowwise().maxCoeff();
    nUser = static_cast<int>(maxVal(0))+1;
    nItem = static_cast<int>(maxVal(1))+1;
    // Each column is a param vector.
    DEBUG_ASSERT(userParam.rows() == FEAT_DIM);
    DEBUG_ASSERT(itemParam.rows() == FEAT_DIM);
    // userParam.resize(userParam.rows(), nUser);
    // itemParam.resize(itemParam.rows(), nItem);
    // Random(): For floating points, uniform distribution [-1, 1]
    userParam = ParamType::Random(userParam.rows(), nUser);
    itemParam = ParamType::Random(itemParam.rows(), nItem);
    perm.resize(ratings.cols());
    std::iota(perm.begin(), perm.end(), 0);

    numUserRating.assign(nUser, 0);
    numItemRating.assign(nItem, 0);
    for (int dataId = 0; dataId < nTrain; dataId++) {
      int user = static_cast<int>(ratings(0, dataId));
      int item = static_cast<int>(ratings(1, dataId));
      numUserRating[user]++;
      numItemRating[item]++;
    }
  }

  void Iterate() {
    // Iterate
    // TODO: set a random seed

    // Random shuffle.
    std::random_shuffle(perm.begin(), perm.end());
    DEBUG_ASSERT(perm.size() == ratings.cols());

    // LOG("Performing %ld iterations\n", ratings.cols());
    for (int i = 0; i < perm.size(); i++) {
      int dataId = perm[i];
      int user = static_cast<int>(ratings(0, dataId));
      int item = static_cast<int>(ratings(1, dataId));
      double rating = ratings(2, dataId);
      
      // gradient
      double diff = rating - userParam.col(user).dot(itemParam.col(item));
      VectorXd userGrad = -2.0 * diff * itemParam.col(item) +
          2.0 * lambda * userParam.col(user) / numUserRating[user];
      VectorXd itemGrad = -2.0 * diff * userParam.col(user) +
          2.0 * lambda * itemParam.col(item) / numItemRating[item];
      
      // update
      userParam.col(user) -= stepSize * userGrad;
      itemParam.col(item) -= stepSize * itemGrad;
      // if (i % 100000 == 0) {
      //   LOG("Finished %d iterations\n", i);
      // }
    }
  }

  double ComputeLoss() {
    const int nTrain = ratings.cols();
    double loss = 0;
    // prediction term
    for (int i = 0; i < nTrain; i++) {
      int user = static_cast<int>(ratings(0, i));
      int item = static_cast<int>(ratings(1, i));
      double rating = ratings(2, i);
      double diff = rating - userParam.col(user).dot(itemParam.col(item));
      loss += diff * diff;

      // regularization term
      // loss += lambda * userParam.col(user).squaredNorm();
      // loss += lambda * itemParam.col(item).squaredNorm();
    }
    loss /= nTrain;

    return loss;
  }
};

int main(int argc, char **argv) {
  // const std::string trainFile = "../../../ml-100k/u1.base";
  // DataReader100K dataReader(trainFile);

  const std::string trainFile = "../../../data/train_shuffle.dat";
  DataReader10M trainDataReader(trainFile);

  const std::string testFile = "../../../data/test_shuffle.dat";
  DataReader10M testDataReader(testFile);

  const double stepSize = 0.001;
  const double lambda = 1.0;
  const int nEpoch = 10;
  SGD sgd(trainDataReader.MoveData(), lambda, stepSize, nEpoch);
  sgd.Train();
  std::cout << "rmse = " << sgd.TestRMSE(testDataReader.Data()) << std::endl;
}
