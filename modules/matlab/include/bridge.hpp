#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"
#include <vector>
#include <opencv2/core.hpp>
#include <ext/hash_map>

/* 
 * Custom typedefs
 * Parsed names from the hdr_parser
 */
typedef std::vector<cv::Mat> vector_Mat;
typedef std::vector<cv::Point> vector_Point;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<unsigned char> vector_uchar;


void conditionalError(bool expr, const std::string& str) {
  if (!expr) mexErrMsgTxt(std::string("condition failed: ").append(str).c_str());
}

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
private:
  mxArray* ptr_;
public:
  // bridges are default constructible
  Bridge() {}
  virtual ~Bridge() {}

  /*! @brief unpack an object from Matlab into C++
   *
   * this function checks whether the given bridge is derived from an
   * object in Matlab. If so, it converts it to a (platform dependent)
   * pointer to the underlying C++ object.
   *
   * NOTE! This function assumes that the C++ pointer is stored in inst_
   */
  template <typename Object>
  Object* getObjectByName(const char* name) {
    // check that the object is actually of correct type before unpacking
    // TODO: Traverse class hierarchy?
    if (!mxIsClass(ptr_, name)) {
      const char* actual = mxGetClassName(ptr_);
      mexErrMsgTxt(std::string("Expected class ").append(std::string(name))
                       .append(" but was given ").append(std::string(actual)).c_str());
    }
    // get the instance field
    mxArray* inst = mxGetField(ptr_, 0, "inst_");
    Object* obj = NULL;
    // make sure the pointer is the correct size for the system
    if (sizeof(void *) == 8 && mxIsClass(inst, "uint64")) {
      // 64-bit pointers
      // TODO: Do we REALLY REALLY need to reinterpret_cast?
      obj = reinterpret_cast<Object *>(
            reinterpret_cast<uint64_t *>(mxGetData(inst))[0]);
    } else if (sizeof(void *) == 4 && mxIsClass(inst, "uint32")) {
      // 32-bit pointers
      obj = reinterpret_cast<Object *>(
            reinterpret_cast<uint32_t *>(mxGetData(inst))[0]);
    } else {
      mexErrMsgTxt("Incorrect pointer type stored for architecture");
    }

    // finally check if the object is NULL
    if (!obj) mexErrMsgTxt(std::string("Object ").append(std::string(name)).append(std::string(" is NULL")).c_str());
    return obj;
  }

  // --------------------------- mxArray --------------------------------------
  Bridge& operator=(const mxArray* obj) { return *this; }
  Bridge(const mxArray* obj) {}
  mxArray* toMxArray() { return NULL; }

  // --------------------------- cv::Mat --------------------------------------
  Bridge& operator=(const cv::Mat& obj) { return *this; }
  cv::Mat toMat() { return cv::Mat(); }
  operator cv::Mat() { return toMat(); }

  // -------------------- vector_Mat --------------------------------
  Bridge& operator=(const vector_Mat& obj) { return *this; }
  vector_Mat toVectorMat() { return vector_Mat(); }
  operator vector_Mat() { return toVectorMat(); }

  // ---------------------------   int   --------------------------------------
  Bridge& operator=(const int& obj) { return *this; }
  int toInt() { return 0; }
  operator int() { return toInt(); }

  // --------------------------- vector_int  ----------------------------------
  Bridge& operator=(const vector_int& obj) { return *this; }
  vector_int toVectorInt() { return vector_int(); }
  operator vector_int() { return toVectorInt(); }

  // --------------------------- vector_float  ----------------------------------
  Bridge& operator=(const vector_float& obj) { return *this; }
  vector_float toVectorFloat() { return vector_float(); }
  operator vector_float() { return toVectorFloat(); }

  // --------------------------- string  --------------------------------------
  Bridge& operator=(const std::string& obj) { return *this; }
  std::string toString() { return ""; }
  operator std::string() { return toString(); }

  // ---------------------------  bool   --------------------------------------
  Bridge& operator=(const bool& obj) { return *this; }
  bool toBool() { return 0; }
  operator bool() { return toBool(); }

  // --------------------------- double  --------------------------------------
  Bridge& operator=(const double& obj) { return *this; }
  double toDouble() { return 0; }
  operator double() { return toDouble(); }

  // --------------------------   Point  --------------------------------------
  Bridge& operator=(const cv::Point& obj) { return *this; }
  cv::Point toPoint() { return cv::Point(); }
  operator cv::Point() { return toPoint(); }

  // --------------------------   Point2f  ------------------------------------
  Bridge& operator=(const cv::Point2f& obj) { return *this; }
  cv::Point2f toPoint2f() { return cv::Point2f(); }
  operator cv::Point2f() { return toPoint2f(); }

  // --------------------------   Point2d  ------------------------------------
  Bridge& operator=(const cv::Point2d& obj) { return *this; }
  cv::Point2d toPoint2d() { return cv::Point2d(); }
  operator cv::Point2d() { return toPoint2d(); }

  // --------------------------   Size  ---------------------------------------
  Bridge& operator=(const cv::Size& obj) { return *this; }
  cv::Size toSize() { return cv::Size(); }
  operator cv::Size() { return toSize(); }

  // -------------------------- Moments  ---------------------------------------
  Bridge& operator=(const cv::Moments& obj) { return *this; }
  cv::Moments toMoments() { return cv::Moments(); }
  operator cv::Moments() { return toMoments(); }

  // ------------------------ vector_Point ------------------------------------
  Bridge& operator=(const vector_Point& obj) { return *this; }
  vector_Point toVectorPoint() { return vector_Point(); }
  operator vector_Point() { return toVectorPoint(); }

  // ------------------------ vector_uchar -------------------------------------
  Bridge& operator=(const vector_uchar& obj) { return *this; }
  vector_uchar toVectorUchar() { return vector_uchar(); }
  operator vector_uchar() { return toVectorUchar(); }

  // --------------------------  Scalar  --------------------------------------
  Bridge& operator=(const cv::Scalar& obj) { return *this; }
  cv::Scalar toScalar() { return cv::Scalar(); }
  operator cv::Scalar() { return toScalar(); }

  // -------------------------- Rect  --------------------------------------
  Bridge& operator=(const cv::Rect& obj) { return *this; }
  cv::Rect toRect() { return cv::Rect(); }
  operator cv::Rect() { return toRect(); }

  // ---------------------- RotatedRect ------------------------------------
  Bridge& operator=(const cv::RotatedRect& obj) { return *this; }
  cv::RotatedRect toRotatedRect() { return cv::RotatedRect(); }
  operator cv::RotatedRect() { return toRotatedRect(); }

  // ---------------------- TermCriteria -----------------------------------
  Bridge& operator=(const cv::TermCriteria& obj) { return *this; }
  cv::TermCriteria toTermCriteria() { return cv::TermCriteria(); }
  operator cv::TermCriteria() { return toTermCriteria(); }

  // ----------------------      RNG     -----------------------------------
  Bridge& operator=(const cv::RNG& obj) { return *this; }
  /*! @brief explicit conversion to cv::RNG()
   *
   * Converts a bridge object to a cv::RNG(). We explicitly assert that
   * the object is an RNG in matlab space before attempting to deference
   * its pointer
   */
  cv::RNG toRNG() { 
    return (*getObjectByName<cv::RNG>("RNG"));
  }
  operator cv::RNG() { return toRNG(); }

};

#endif
