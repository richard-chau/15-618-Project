#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <numeric>
#include <cmath>
#include "cf_sgd.h"
#include "CycleTimer.h"

#include <mpi.h>

#define TRAIN_DATA_TAG 0
#define RECV_PARAM_TAG 1
#define SEND_PARAM_TAG 2
#define TEST_DATA_TAG 4


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

class ParallelSGD {
 public:
  ParallelSGD(const std::string& trainFile,
              const std::string& testFile,
              double lambda_,
              double stepSize_,
              int nEpoch_,
              int workerId_,
              int nWorker_):
  lambda(lambda_), stepSize(stepSize_), nEpoch(nEpoch_),
  workerId(workerId_), nWorker(nWorker_) {
    int e; // return value of MPI function.
    master = (workerId == 0);
    // Initialize data.
    if (master) {
      // master reads data.
      // TODO: checks the train data and test data.
      // TODO: the user id or item id may not be continuous.
      DataReader10M dr(trainFile);
      ratings = std::move(dr.MoveData());

      DataReader10M drTest(testFile);
      testRatings = std::move(drTest.MoveData());
      
      // check if the ratings are sorted by user id.
      #ifndef NDEBUG
        LOG("Checks ordering of training data.\n");
        for (int i = 1; i < ratings.cols(); i++) {
          int prevUser = static_cast<int>(ratings(0, i-1));
          int curUser = static_cast<int>(ratings(0, i));
          assert(prevUser <= curUser);
        }
        LOG("checks ordering of test data.\n");
        for (int i = 1; i < testRatings.cols(); i++) {
          int prevUser = static_cast<int>(testRatings(0, i-1));
          int curUser = static_cast<int>(testRatings(0, i));
          assert(prevUser <= curUser);
        }
      #endif
      
      nTrain = static_cast<int>(ratings.cols());
      nTest = static_cast<int>(testRatings.cols());
    }
    // send data size
    int buf[4];
    if (master) {
      Eigen::VectorXd maxVal = ratings.rowwise().maxCoeff();
      nUser = static_cast<int>(maxVal(0))+1;
      nItem = static_cast<int>(maxVal(1))+1;
      buf[0] = nTrain;
      buf[1] = nUser;
      buf[2] = nItem;
      buf[3] = nTest;
    }
    e = MPI_Bcast(buf, 4, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    if (!master) {
      nTrain = buf[0];
      nUser = buf[1];
      nItem = buf[2];
      nTest = buf[3];
    }

    LOG("nTrain=%d, nUser=%d, nItem=%d, nTest=%d\n", nTrain, nUser, nItem, nTest);

    // compute numUserRating and numItemRating.
    numUserRating.assign(nUser, 0);
    numItemRating.assign(nItem, 0);
    if (master) {
      for (int dataId = 0; dataId < nTrain; dataId++) {
        int user = static_cast<int>(ratings(0, dataId));
        int item = static_cast<int>(ratings(1, dataId));
        numUserRating[user]++;
        numItemRating[item]++;
      }

      // #ifndef NDEBUG
      //   LOG("Check numUserRating and numItemRating\n");
      //   for (int i = 0; i < nUser; i++) {
      //     LOG("%d", i);
      //     DEBUG_ASSERT(numUserRating[i] > 0);
      //   }
      //   for (int i = 0; i < nItem; i++) {
      //     DEBUG_ASSERT(numItemRating[i] > 0);
      //   }
      // #endif
    }
    e = MPI_Bcast(numUserRating.data(), nUser, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    e = MPI_Bcast(numItemRating.data(), nItem, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);


    // compute userRanges
    userRanges.resize(nWorker);
    for (int wid = 0; wid < nWorker; wid++) {
      int start = wid * (nUser/nWorker), end;
      if (wid == nWorker - 1) end = nUser;
      else end = (wid+1) * (nUser/nWorker);
      userRanges[wid] = {start, end};
    }

    
    // Send test data, store in localTestRatings.
    if (master) {
      MPI_Request *sendRequests = new MPI_Request[nWorker-1];
      int sendReqCnt = 0;
      double* p = testRatings.data();
      int prevPos = 0;
      for (int wid = 0; wid < nWorker; wid++) {
        int curPos = -1;
        if (wid == nWorker - 1) {
          curPos = testRatings.cols();
        } else {
          int endUser = userRanges[wid].second;
          for (int j = prevPos; j < testRatings.cols(); j++) {
            if (static_cast<int>(testRatings(0, j)) >= endUser) {
              curPos = j;
              break;
            }
          }
        }
        DEBUG_ASSERT(curPos != -1);

        if (wid == 0) {
          localTestRatings = testRatings.block(0, 0, 3, curPos-prevPos);
        } else {
          double* start = p + 3 * prevPos;
          int size = 3 * (curPos - prevPos);
          e = MPI_Isend(start, size, MPI_DOUBLE, wid, TEST_DATA_TAG,
              MPI_COMM_WORLD, &sendRequests[sendReqCnt++]);
          DEBUG_ASSERT(e == MPI_SUCCESS);
        }
        prevPos = curPos;
      }
      DEBUG_ASSERT(sendReqCnt == nWorker-1);

      e = MPI_Waitall(sendReqCnt, sendRequests, MPI_STATUSES_IGNORE);
      DEBUG_ASSERT(e == MPI_SUCCESS);
      delete [] sendRequests;

      testRatings.resize(0, 0);
    } else {
      MPI_Status status;
      MPI_Probe(0, TEST_DATA_TAG, MPI_COMM_WORLD, &status);
      int cnt;
      MPI_Get_count(&status, MPI_DOUBLE, &cnt);
      DEBUG_ASSERT(cnt % 3 == 0);
      localTestRatings.resize(3, cnt / 3);
      e = MPI_Recv(localTestRatings.data(), cnt, MPI_DOUBLE,
          0, TEST_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      DEBUG_ASSERT(e == MPI_SUCCESS);
    }
    LOG("%d: local test data size = %d\n", workerId, localTestRatings.cols());


    // Send training data, store in localRatings.
    if (master) {
      MPI_Request *sendRequests = new MPI_Request[nWorker-1];
      int sendReqCnt = 0;
      double* p = ratings.data();
      int prevPos = 0;
      for (int wid = 0; wid < nWorker; wid++) {
        int curPos = -1;
        if (wid == nWorker - 1) {
          curPos = ratings.cols();
        } else {
          int endUser = userRanges[wid].second;
          for (int j = prevPos; j < ratings.cols(); j++) {
            if (static_cast<int>(ratings(0, j)) >= endUser) {
              curPos = j;
              break;
            }
          }
        }
        DEBUG_ASSERT(curPos != -1);

        if (wid == 0) {
          localRatings = ratings.block(0, 0, 3, curPos-prevPos);
        } else {
          double* start = p + 3 * prevPos;
          int size = 3 * (curPos - prevPos);
          e = MPI_Isend(start, size, MPI_DOUBLE, wid, TRAIN_DATA_TAG,
              MPI_COMM_WORLD, &sendRequests[sendReqCnt++]);
          DEBUG_ASSERT(e == MPI_SUCCESS);
        }
        prevPos = curPos;
      }
      DEBUG_ASSERT(sendReqCnt == nWorker-1);

      e = MPI_Waitall(sendReqCnt, sendRequests, MPI_STATUSES_IGNORE);
      DEBUG_ASSERT(e == MPI_SUCCESS);
      delete [] sendRequests;

      ratings.resize(0, 0);
    } else {
      MPI_Status status;
      MPI_Probe(0, TRAIN_DATA_TAG, MPI_COMM_WORLD, &status);
      int cnt;
      MPI_Get_count(&status, MPI_DOUBLE, &cnt);
      DEBUG_ASSERT(cnt % 3 == 0);
      localRatings.resize(3, cnt / 3);
      e = MPI_Recv(localRatings.data(), cnt, MPI_DOUBLE,
          0, TRAIN_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      DEBUG_ASSERT(e == MPI_SUCCESS);
    }
    LOG("%d: local data size = %d\n", workerId, localRatings.cols());
    // LOG("(%lf, %lf, %lf)\n", localRatings(0, 0),
    //     localRatings(1, 0), localRatings(2, 0));
    

    // sort localRatings by item id and check.
    sortByItemId();
    #ifndef NDEBUG
      LOG("Checks ordering of local training data after sortByItemId()\n");
        for (int i = 1; i < localRatings.cols(); i++) {
          int prevItem = static_cast<int>(localRatings(1, i-1));
          int curItem = static_cast<int>(localRatings(1, i));
          assert(prevItem <= curItem);
        }
    #endif
    LOG("%d: sort by item id\n", workerId);

    // compute ItemRanges
    itemRanges.resize(nWorker);
    for (int strataId = 0; strataId < nWorker; strataId++) {
      int start = strataId * (nItem/nWorker), end;
      if (strataId == nWorker - 1) end = nItem;
      else end = (strataId+1) * (nItem/nWorker);
      itemRanges[strataId] = {start, end};
    }


    // Organize data based on starta.
    strata.resize(nWorker);
    int prevStrataEnd = 0;
    for (int strataId = 0; strataId < nWorker; strataId++) {
      int curStrataEnd = -1;
      if (strataId == nWorker - 1) {
        curStrataEnd = localRatings.cols();
      } else {
        int endItem = itemRanges[strataId].second;
        for (int j = prevStrataEnd; j < localRatings.cols(); j++) {
          if (localRatings(1, j) >= endItem) {
            curStrataEnd = j;
            break;
          }
        }
        if (curStrataEnd == -1)
          curStrataEnd = localRatings.cols();
      }
      strata[strataId].resize(curStrataEnd - prevStrataEnd);
      std::iota(strata[strataId].begin(), strata[strataId].end(), prevStrataEnd);
      prevStrataEnd = curStrataEnd;
    }
    // check strata covers all local ratings.
    #ifndef NDEBUG
      LOG("%d: check strata.\n", workerId);
      int ind = 0;
      for (const auto& vec : strata) {
        for (int index : vec) {
          assert(ind == index);
          ind++;
        }
      }
      assert(ind == localRatings.cols());
    #endif
  }

  void Train() {
    InitializeParam();
    double ltrain, ltest;
    std::tie(ltrain, ltest) = ComputeRMSE();
    LOG("Initial RMSE = %lf, %lf \n", ltrain, ltest);
    record(ltrain, ltest, 0, 0, 0);
    for (int epoch = 0; epoch < nEpoch; epoch++) {
      // LOG("Start epoch %d\n", epoch);
      double commTime = 0.0;
      double iterTime = 0.0;
      double lossTime = 0.0;
      for (int subEpoch = 0; subEpoch < nWorker; subEpoch++) {
        int strataId = (workerId + subEpoch) % nWorker;
        double t1 = CycleTimer::currentSeconds();
        if (master) MasterSendParam(strataId);
        else ReceiveParam(strataId);
        double t2 = CycleTimer::currentSeconds();
        commTime += t2 - t1;
        // LOG("workerid %d: Start subepoch %d %d\n",
        //       workerId, subEpoch, strata[strataId].size());
        Iterate(strataId);
        double t3 = CycleTimer::currentSeconds();
        iterTime += t3 - t2;
        if (master) MasterReceiveParam(strataId);
        else SendParam(strataId);
        double t4 = CycleTimer::currentSeconds();
        commTime += t4 - t3;
      }
      // TODO: compute loss.
      double t5 = CycleTimer::currentSeconds();
      // only totalLoss at master node is valid.
      double trainrmse, testrmse;
      std::tie(trainrmse, testrmse) = ComputeRMSE();
      double t6 = CycleTimer::currentSeconds();
      lossTime += t6 - t5;
      if (master) {
        
        // Note: RMSE error
        // totalLoss = totalLoss;
        // losses.emplace_back(totalLoss);
        LOG("Epoch %d loss=%lf, %lf, iteration time=%lf, loss computation time=%lf, commTime=%lf\n",
          epoch, trainrmse, testrmse, iterTime, lossTime, commTime);
        record(trainrmse, testrmse, iterTime, lossTime, commTime);
      }
    }
    if (master) {
      PrintResult();
    }
  }

 private:
  double lambda; // regularization term
  double stepSize;
  int nEpoch;
  int workerId;
  int nWorker;
  int nTrain;
  int nUser;
  int nItem;
  int nTest;

  bool master;

  DataType ratings;
  DataType localRatings;
  ParamType userParam;
  ParamType itemParam;

  DataType testRatings;
  DataType localTestRatings;

  std::vector<std::vector<int>> strata;

  std::vector<int> numUserRating;
  std::vector<int> numItemRating;

  std::vector<std::pair<int, int>> userRanges; // user ranges in each worker
  std::vector<std::pair<int, int>> itemRanges; // item range in each strata

  std::vector<double> losses;
  std::vector<double> trainRMSE;
  std::vector<double> testRMSE;
  std::vector<double> trainTime;
  std::vector<double> rmseCompTime;
  std::vector<double> commTime;

  void record(double trainErr, double testErr, double trainT,
              double rmseT, double cTime) {
    trainRMSE.emplace_back(trainErr);
    testRMSE.emplace_back(testErr);
    trainTime.emplace_back(trainT);
    rmseCompTime.emplace_back(rmseT);
    commTime.emplace_back(cTime);
  }

  class Comparator {
   public:
    Comparator(const DataType& r_) : r(r_) { }
    bool operator () (int i, int j) {
      int iItem = static_cast<int>(r(1, i));
      int jItem = static_cast<int>(r(1, j));
      int iUser = static_cast<int>(r(0, i));
      int jUser = static_cast<int>(r(0, j));
      return iItem < jItem || (iItem == jItem && iUser <= jUser);
    }
   private:
    const DataType& r;
  };

  void sortByItemId() {
    const int nLocalRatings = localRatings.cols();
    std::vector<int> indices(localRatings.cols());
    std::iota(indices.begin(), indices.end(), 0);
    // std::sort(indices.begin(), indices.end(),
    //     [&](int i, int j) {
    //       int iItem = static_cast<int>(localRatings(1, i));
    //       int jItem = static_cast<int>(localRatings(1, j));
    //       int iUser = static_cast<int>(localRatings(0, i));
    //       int jUser = static_cast<int>(localRatings(0, j));
    //       return iItem < jItem || (iItem == jItem && iUser <= jUser);
    // });
    std::sort(indices.begin(), indices.end(), Comparator(localRatings));

    DataType sortedLocalRatings(localRatings.rows(), localRatings.cols());
    for (int i = 0; i < indices.size(); i++) {
      sortedLocalRatings.col(i) = localRatings.col(indices[i]);
    }
    localRatings = std::move(sortedLocalRatings);
    DEBUG_ASSERT(nLocalRatings == localRatings.cols());
  }

  void InitializeParam() {
    // Initialize parameter.
    // Each column is a param vector.
    DEBUG_ASSERT(userParam.rows() == FEAT_DIM);
    DEBUG_ASSERT(itemParam.rows() == FEAT_DIM);
    // Random(): For floating points, uniform distribution [-1, 1]
    userParam = ParamType::Random(userParam.rows(), nUser);
    itemParam = ParamType::Random(itemParam.rows(), nItem);
    // perm.resize(ratings.cols());
    // std::iota(perm.begin(), perm.end(), 0);
  }

  void Iterate(int strataId) {
    // Iterate
    // TODO: set a random seed

    // Random shuffle.
    std::vector<int>& dataIds = strata[strataId];
    std::random_shuffle(dataIds.begin(), dataIds.end());

    for (int i = 0; i < dataIds.size(); i++) {
      int dataId = dataIds[i];
      int user = static_cast<int>(localRatings(0, dataId));
      int item = static_cast<int>(localRatings(1, dataId));
      double rating = localRatings(2, dataId);
      DEBUG_ASSERT(user >= userRanges[workerId].first &&
                    user < userRanges[workerId].second);
      DEBUG_ASSERT(item >= itemRanges[strataId].first &&
                    item < itemRanges[strataId].second);
      DEBUG_ASSERT(rating >= 0.5 && rating <= 5.0);

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

  void ReceiveParam(int strataId) {
    int startItem = itemRanges[strataId].first;
    int endItem = itemRanges[strataId].second;
    int size = itemParam.rows() * (endItem - startItem);
    double* startAddr = itemParam.data() + startItem * itemParam.rows();
    MPI_Status status;
    int e = MPI_Recv(startAddr, size, MPI_DOUBLE, 0, RECV_PARAM_TAG,
                      MPI_COMM_WORLD, &status);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    int recvCnt;
    MPI_Get_count(&status, MPI_DOUBLE, &recvCnt);
    DEBUG_ASSERT(recvCnt == size);
  }

  void SendParam(int strataId) {
    int startItem = itemRanges[strataId].first;
    int endItem = itemRanges[strataId].second;
    int size = itemParam.rows() * (endItem - startItem);
    double* startAddr = itemParam.data() + startItem * itemParam.rows();
    int e = MPI_Send(startAddr, size, MPI_DOUBLE, 0,
                      SEND_PARAM_TAG, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);
  }

  void MasterReceiveParam(int strataId) {
    MPI_Request *recvRequests = new MPI_Request[nWorker-1];
    MPI_Status *statuses = new MPI_Status[nWorker-1];
    int recvReqCnt = 0;
    for (int wid = 1; wid < nWorker; wid++) {
      int sId = (strataId + wid) % nWorker;
      int startItem = itemRanges[sId].first;
      int endItem = itemRanges[sId].second;
      int size = itemParam.rows() * (endItem - startItem);
      double* startAddr = itemParam.data() + startItem * itemParam.rows();
      int e = MPI_Irecv(startAddr, size, MPI_DOUBLE, wid, SEND_PARAM_TAG,
                        MPI_COMM_WORLD, &recvRequests[recvReqCnt++]);
      DEBUG_ASSERT(e == MPI_SUCCESS);
    }
    DEBUG_ASSERT(recvReqCnt == nWorker-1);

    int e = MPI_Waitall(recvReqCnt, recvRequests, statuses);
    DEBUG_ASSERT(e == MPI_SUCCESS);

    for (int wid = 1; wid < nWorker; wid++) {
      int sId = (strataId + wid) % nWorker;
      int startItem = itemRanges[sId].first;
      int endItem = itemRanges[sId].second;
      int size = itemParam.rows() * (endItem - startItem);
      int recvCnt;
      MPI_Get_count(&statuses[wid-1], MPI_DOUBLE, &recvCnt);
      DEBUG_ASSERT(recvCnt == size);
    }

    delete [] recvRequests;
    delete [] statuses;
  }

  void MasterSendParam(int strataId) {
    MPI_Request *sendRequests = new MPI_Request[nWorker-1];
    int sendReqCnt = 0;
    for (int wid = 1; wid < nWorker; wid++) {
      int sId = (strataId + wid) % nWorker;
      int startItem = itemRanges[sId].first;
      int endItem = itemRanges[sId].second;
      int size = itemParam.rows() * (endItem - startItem);
      double* startAddr = itemParam.data() + startItem * itemParam.rows();
      int e = MPI_Isend(startAddr, size, MPI_DOUBLE, wid, RECV_PARAM_TAG,
                        MPI_COMM_WORLD, &sendRequests[sendReqCnt++]);
      DEBUG_ASSERT(e == MPI_SUCCESS);
    }

    int e = MPI_Waitall(sendReqCnt, sendRequests, MPI_STATUSES_IGNORE);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    delete [] sendRequests;
  }

  std::pair<double, double> ComputeRMSE() {
    // master broadcast itemParam to all workers.
    int size = itemParam.rows() * itemParam.cols();
    int e;
    e = MPI_Bcast(itemParam.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);

    // Each worker computes loss w.r.t its local training data.
    double loss = 0.0;
    // TODO: impl this.
    for (int i = 0; i < localRatings.cols(); i++) {
      int user = static_cast<int>(localRatings(0, i));;
      int item = static_cast<int>(localRatings(1, i));
      double rating = localRatings(2, i);
      double diff = rating - userParam.col(user).dot(itemParam.col(item));
      loss += diff * diff;

      // regularization term
      // loss += lambda * userParam.col(user).squaredNorm();
      // loss += lambda * itemParam.col(item).squaredNorm();
    }

    // Reduce to master.
    double totalLoss;
    e = MPI_Reduce(&loss, &totalLoss, 1, MPI_DOUBLE,
                    MPI_SUM, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    double trainrmse = std::sqrt(totalLoss / nTrain);

    // testRMSE
    double testLoss = 0.0;
    for (int i = 0; i < localTestRatings.cols(); i++) {
      int user = static_cast<int>(localTestRatings(0, i));;
      int item = static_cast<int>(localTestRatings(1, i));
      double rating = localTestRatings(2, i);
      double diff = rating - userParam.col(user).dot(itemParam.col(item));
      testLoss += diff * diff;
    }
    double totalTestLoss;
    e = MPI_Reduce(&testLoss, &totalTestLoss, 1, MPI_DOUBLE,
                    MPI_SUM, 0, MPI_COMM_WORLD);
    DEBUG_ASSERT(e == MPI_SUCCESS);
    double testrmse = std::sqrt(totalTestLoss / nTest);

    return {trainrmse, testrmse};
  }

  void PrintVector(const std::vector<double>& vec) {
    std::cout << "[ ";
    for (int i = 0; i < vec.size(); i++) {
      if (i != vec.size() - 1)
        std::cout << vec[i] << ", ";
      else
        std::cout << vec[i] << " ]" << std::endl;
    }
  }

  void PrintResult() {
    PrintVector(trainRMSE);
    PrintVector(testRMSE);
    PrintVector(trainTime);
    PrintVector(rmseCompTime);
    PrintVector(commTime);
  }
};

int main(int argc, char *argv[]) {
  int workerId, nWorker;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nWorker);
  MPI_Comm_rank(MPI_COMM_WORLD, &workerId);
  LOG("%d workders\n", nWorker);

  const std::string trainFile = "/home/wenhaoh/FinalProject/data/train_shuffle.dat";
  const std::string testFile = "/home/wenhaoh/FinalProject/data/test_shuffle.dat";
  const double stepSize = 0.001;
  const double lambda = 1.0;
  int nEpoch = 40;
  std::cout << "nWorker = " << nWorker << std::endl;
  if (nWorker == 1) {
    nEpoch = 10;
  } else if (nWorker == 4) {
    nEpoch = 40;
  } else if (nWorker == 6) {
    nEpoch = 60;
  } else if (nWorker == 12) {
    nEpoch = 80;
  } else {
    std::cerr << "wrong config for nWorker\n" << std::endl; 
    exit(1);
  }

  ParallelSGD sgd(trainFile, testFile, lambda, stepSize, nEpoch, workerId, nWorker);
  sgd.Train();
  int e = MPI_Finalize();
  DEBUG_ASSERT(e == MPI_SUCCESS);
}
