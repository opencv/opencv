#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"
#include <opencv2/core.hpp>

/*! 
 * @class Bridge
 * @brief Type conversion class for converting OpenCV and native C++ types
 *
 * Bridge provides an interface for converting between OpenCV/C++ types
 * to Matlab's mxArray format.
 *
 * Each type conversion requires three operators:
 *    // conversion from ObjectType --> Bridge
 *    Bridge& operator=(const ObjectType&);
 *    // implicit conversion from Bridge --> ObjectType
 *    operator ObjectType();
 *    // explicit conversion from Bridge --> ObjectType
 *    ObjectType toObjectType
 *
 * The bridging class provides common conversions between OpenCV types,
 * std and stl types to Matlab's mxArray format. By inheriting Bridge, 
 * you can add your own custom type conversions.
 *
 * Bridge attempts to make as few assumptions as possible, however in 
 * some cases where 1-to-1 mappings don't exist, some assumptions are necessary.
 * In particular:
 *  - conversion from of a 2-channel Mat to an mxArray will result in a complex
 *    output
 *  - conversion from multi-channel interleaved Mats will result in
 *    multichannel planar mxArrays
 *
 */
class Bridge {
public:
  // bridges are default constructible
  Bridge() {}
  virtual ~Bridge() {}

  // --------------------------- mxArray --------------------------------------
  Bridge& operator=(const mxArray* obj) {}
  Bridge(const mxArray* obj) {}
  mxArray* mxArray() { return NULL; }

  // --------------------------- cv::Mat --------------------------------------
  Bridge& operator=(const cv::Mat& obj) {}
  operator cv::Mat() { return cv::Mat(); }
  cv::Mat toMat() { return cv::Mat(); }

  // ---------------------------   int   --------------------------------------
  Bridge& operator=(const int& obj) {}
  operator int() { return 0; }
  int toInt() { return 0; }
};

#endif
