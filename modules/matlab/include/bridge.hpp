#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"
#include <vector>
#include <opencv2/core.hpp>

/* 
 * Custom typedefs
 * Parsed names from the hdr_parser
 */
typedef std::vector<cv::Mat> vector_Mat;
typedef std::vector<cv::Point> vector_Point;
typedef std::vector<int> vector_int;

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
 *    ObjectType toObjectType();
 *
 * The bridging class provides common conversions between OpenCV types,
 * std and stl types to Matlab's mxArray format. By inheriting Bridge, 
 * you can add your own custom type conversions.
 *
 * NOTE: for the explicit conversion function, the object name must be
 * in UpperCamelCase, for example:
 *    int --> toInt
 *    my_object --> MyObject
 *    my_Object --> MyObject
 *    myObject  --> MyObject
 * this is because the binding generator standardises the calling syntax.
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
  Bridge& operator=(const mxArray* obj) { return *this; }
  Bridge(const mxArray* obj) {}
  mxArray* toMxArray() { return NULL; }

  // --------------------------- cv::Mat --------------------------------------
  Bridge& operator=(const cv::Mat& obj) { return *this; }
  operator cv::Mat() { return cv::Mat(); }
  cv::Mat toMat() { return cv::Mat(); }

  // -------------------- vector_Mat --------------------------------
  Bridge& operator=(const vector_Mat& obj) { return *this; }
  operator vector_Mat() { return vector_Mat(); }
  vector_Mat toVectorMat() { return vector_Mat(); }

  // ---------------------------   int   --------------------------------------
  Bridge& operator=(const int& obj) { return *this; }
  operator int() { return 0; }
  int toInt() { return 0; }

  // --------------------------- vector_int  ----------------------------------
  Bridge& operator=(const vector_int& obj) { return *this; }
  operator vector_int() { return vector_int(); }
  vector_int toVectorInt() { return vector_int(); }

  // --------------------------- string  --------------------------------------
  Bridge& operator=(const std::string& obj) { return *this; }
  operator std::string() { return ""; }
  std::string toString() { return ""; }

  // ---------------------------  bool   --------------------------------------
  Bridge& operator=(const bool& obj) { return *this; }
  operator bool() { return 0; }
  bool toBool() { return 0; }

  // --------------------------- double  --------------------------------------
  Bridge& operator=(const double& obj) { return *this; }
  operator double() { return 0; }
  double toDouble() { return 0; }

  // --------------------------   Point  --------------------------------------
  Bridge& operator=(const cv::Point& obj) { return *this; }
  operator cv::Point() { return cv::Point(); }
  cv::Point toPoint() { return cv::Point(); }

  // --------------------------   Size  ---------------------------------------
  Bridge& operator=(const cv::Size& obj) { return *this; }
  operator cv::Size() { return cv::Size(); }
  cv::Size toSize() { return cv::Size(); }

  // ------------------------ vector_Point ------------------------------------
  Bridge& operator=(const vector_Point& obj) { return *this; }
  operator vector_Point() { return vector_Point(); }
  vector_Point toVectorPoint() { return vector_Point(); }

  // --------------------------  Scalar  --------------------------------------
  Bridge& operator=(const cv::Scalar& obj) { return *this; }
  operator cv::Scalar() { return cv::Scalar(); }
  cv::Scalar toScalar() { return cv::Scalar(); }

  // -------------------------- Rect  --------------------------------------
  Bridge& operator=(const cv::Rect& obj) { return *this; }
  operator cv::Rect() { return cv::Rect(); }
  cv::Rect toRect() { return cv::Rect(); }

  // ---------------------- RotatedRect ------------------------------------
  Bridge& operator=(const cv::RotatedRect& obj) { return *this; }
  operator cv::RotatedRect() { return cv::RotatedRect(); }
  cv::RotatedRect toRotatedRect() { return cv::RotatedRect(); }

  // ---------------------- TermCriteria -----------------------------------
  Bridge& operator=(const cv::TermCriteria& obj) { return *this; }
  operator cv::TermCriteria() { return cv::TermCriteria(); }
  cv::TermCriteria toTermCriteria() { return cv::TermCriteria(); }

  // ----------------------      RNG     -----------------------------------
  Bridge& operator=(const cv::RNG& obj) { return *this; }
  operator cv::RNG() { return cv::RNG(); }
  cv::RNG toRNG() { return cv::RNG(); }

};

#endif
