#pragma once

#include <Eigen/Dense>

// Feature dimension
#define FEAT_DIM 10

#define LOG_ENABLED


using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, ColMajor> DataType;
typedef Matrix<double, FEAT_DIM, Dynamic, ColMajor> ParamType;

#ifdef NDEBUG
  #define DEBUG_ASSERT(statement)
#else
  #define DEBUG_ASSERT(statement) assert(statement);
#endif

/////////////////////// logging ///////////////////////////
#ifdef LOG_ENABLED
  #define LOG(...) printf(__VA_ARGS__)
#else
  #define LOG(...)
#endif

