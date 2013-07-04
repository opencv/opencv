#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mxarray.hpp"
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

/* 
 * Custom typedefs
 * Parsed names from the hdr_parser
 */
typedef std::vector<cv::Mat> vector_Mat;
typedef std::vector<cv::Point> vector_Point;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<cv::String> vector_String;
typedef std::vector<unsigned char> vector_uchar;
typedef std::vector<cv::Rect> vector_Rect;
typedef std::vector<cv::KeyPoint> vector_KeyPoint;
typedef cv::Ptr<cv::StereoBM> Ptr_StereoBM;
typedef cv::Ptr<cv::StereoSGBM> Ptr_StereoSGBM;
typedef cv::Ptr<cv::FeatureDetector> Ptr_FeatureDetector;




// ----------------------------------------------------------------------------
//                                 BRIDGE
// ----------------------------------------------------------------------------

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
 * Because Matlab uses a homogeneous storage type, all operations are provided
 * relative to Matlab's type. That is, Bridge always stores an MxArray object
 * and converts to and from other object types on demand.
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
  MxArray ptr_;
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
  Object* getObjectByName(const std::string& name) {
    // check that the object is actually of correct type before unpacking
    // TODO: Traverse class hierarchy?
    if (!ptr_.isClass(name)) {
      error(std::string("Expected class ").append(std::string(name))
                .append(" but was given ").append(ptr_.className()));
    }
    // get the instance field
    MxArray inst = ptr_.field("inst_");
    Object* obj = NULL;
    // make sure the pointer is the correct size for the system
    if (sizeof(void *) == 8 && inst.ID() == mxUINT64_CLASS) {
      // 64-bit pointers
      // TODO: Do we REALLY REALLY need to reinterpret_cast?
      obj = reinterpret_cast<Object *>(inst.scalar<uint64_t>());
    } else if (sizeof(void *) == 4 && inst.ID() == mxUINT32_CLASS) {
      // 32-bit pointers
      obj = reinterpret_cast<Object *>(inst.scalar<uint32_t>());
    } else {
      error("Incorrect pointer type stored for architecture");
    }

    // finally check if the object is NULL
    conditionalError(obj, std::string("Object ").append(std::string(name)).append(std::string(" is NULL")));
    return obj;
  }
 





  // --------------------------------------------------------------------------
  //                           MATLAB TYPES
  // --------------------------------------------------------------------------
  Bridge& operator=(const mxArray* obj) { ptr_ = obj; return *this; }
  Bridge(const mxArray* obj) : ptr_(obj)  {}
  MxArray toMxArray() { return ptr_; }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                         INTEGRAL TYPES
  // --------------------------------------------------------------------------
  
  // --------------------------- string  --------------------------------------
  Bridge& operator=(const std::string& ) { return *this; }
  std::string toString() { 
    return ptr_.toString();
  }
  operator std::string() { return toString(); }

  // ---------------------------  bool   --------------------------------------
  Bridge& operator=(const bool& ) { return *this; }
  bool toBool() { return 0; }
  operator bool() { return toBool(); }

  // --------------------------- double  --------------------------------------
  Bridge& operator=(const double& ) { return *this; }
  double toDouble() { return ptr_.scalar<double>(); }
  operator double() { return toDouble(); }

  // --------------------------- float  ---------------------------------------
  Bridge& operator=(const float& ) { return *this; }
  float toFloat() { return ptr_.scalar<float>(); }
  operator float() { return toFloat(); }

  // ---------------------------   int   --------------------------------------
  Bridge& operator=(const int& ) { return *this; }
  int toInt() { return ptr_.scalar<int>(); }
  operator int() { return toInt(); }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                       CORE OPENCV TYPES
  // --------------------------------------------------------------------------

  // --------------------------- cv::Mat --------------------------------------
  Bridge& operator=(const cv::Mat& mat) { ptr_ = MxArray::FromMat<Matlab::InheritType>(mat); return *this; }
  cv::Mat toMat() const { return ptr_.toMat<Matlab::InheritType>(); }
  operator cv::Mat() const { return toMat(); }
  
  // --------------------------   Point  --------------------------------------
  Bridge& operator=(const cv::Point& ) { return *this; }
  cv::Point toPoint() const { return cv::Point(); }
  operator cv::Point() const { return toPoint(); }
  
  // --------------------------   Point2f  ------------------------------------
  Bridge& operator=(const cv::Point2f& ) { return *this; }
  cv::Point2f toPoint2f() const { return cv::Point2f(); }
  operator cv::Point2f() const { return toPoint2f(); }
  
  // --------------------------   Point2d  ------------------------------------
  Bridge& operator=(const cv::Point2d& ) { return *this; }
  cv::Point2d toPoint2d() const { return cv::Point2d(); }
  operator cv::Point2d() const { return toPoint2d(); }
  
  // --------------------------   Size  ---------------------------------------
  Bridge& operator=(const cv::Size& ) { return *this; }
  cv::Size toSize() const { return cv::Size(); }
  operator cv::Size() const { return toSize(); }
  
  // -------------------------- Moments  --------------------------------------
  Bridge& operator=(const cv::Moments& ) { return *this; }
  cv::Moments toMoments() const { return cv::Moments(); }
  operator cv::Moments() const { return toMoments(); }
  
  // --------------------------  Scalar  --------------------------------------
  Bridge& operator=(const cv::Scalar& ) { return *this; }
  cv::Scalar toScalar() { return cv::Scalar(); }
  operator cv::Scalar() { return toScalar(); }
  
  // -------------------------- Rect  -----------------------------------------
  Bridge& operator=(const cv::Rect& ) { return *this; }
  cv::Rect toRect() { return cv::Rect(); }
  operator cv::Rect() { return toRect(); }
  
  // ---------------------- RotatedRect ---------------------------------------
  Bridge& operator=(const cv::RotatedRect& ) { return *this; }
  cv::RotatedRect toRotatedRect() { return cv::RotatedRect(); }
  operator cv::RotatedRect() { return toRotatedRect(); }
  
  // ---------------------- TermCriteria --------------------------------------
  Bridge& operator=(const cv::TermCriteria& ) { return *this; }
  cv::TermCriteria toTermCriteria() { return cv::TermCriteria(); }
  operator cv::TermCriteria() { return toTermCriteria(); }
  
  // ----------------------      RNG     --------------------------------------
  Bridge& operator=(const cv::RNG& ) { return *this; }
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

  
  
  

  // --------------------------------------------------------------------------
  //                       OPENCV VECTOR TYPES
  // --------------------------------------------------------------------------
  
  // -------------------- vector_Mat ------------------------------------------
  Bridge& operator=(const vector_Mat& ) { return *this; }
  vector_Mat toVectorMat() { return vector_Mat(); }
  operator vector_Mat() { return toVectorMat(); }

  // --------------------------- vector_int  ----------------------------------
  Bridge& operator=(const vector_int& ) { return *this; }
  vector_int toVectorInt() { return vector_int(); }
  operator vector_int() { return toVectorInt(); }
  
  // --------------------------- vector_float  --------------------------------
  Bridge& operator=(const vector_float& ) { return *this; }
  vector_float toVectorFloat() { return vector_float(); }
  operator vector_float() { return toVectorFloat(); }
  
  // --------------------------- vector_Rect  ---------------------------------
  Bridge& operator=(const vector_Rect& ) { return *this; }
  vector_Rect toVectorRect() { return vector_Rect(); }
  operator vector_Rect() { return toVectorRect(); }
  
  // --------------------------- vector_KeyPoint  -----------------------------
  Bridge& operator=(const vector_KeyPoint& ) { return *this; }
  vector_KeyPoint toVectorKeyPoint() { return vector_KeyPoint(); }
  operator vector_KeyPoint() { return toVectorKeyPoint(); }
  
  // --------------------------- vector_String  -------------------------------
  Bridge& operator=(const vector_String& ) { return *this; }
  vector_String toVectorString() { return vector_String(); }
  operator vector_String() { return toVectorString(); }
  
  // ------------------------ vector_Point ------------------------------------
  Bridge& operator=(const vector_Point& ) { return *this; }
  vector_Point toVectorPoint() { return vector_Point(); }
  operator vector_Point() { return toVectorPoint(); }
  
  // ------------------------ vector_uchar ------------------------------------
  Bridge& operator=(const vector_uchar& ) { return *this; }
  vector_uchar toVectorUchar() { return vector_uchar(); }
  operator vector_uchar() { return toVectorUchar(); }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                       OPENCV COMPOUND TYPES
  // --------------------------------------------------------------------------

  // ---------------------------   Ptr_StereoBM   -----------------------------
  Bridge& operator=(const Ptr_StereoBM& ) { return *this; }
  Ptr_StereoBM toPtrStereoBM() { return Ptr_StereoBM(); }
  operator Ptr_StereoBM() { return toPtrStereoBM(); }

  // ---------------------------   Ptr_StereoSGBM   ---------------------------
  Bridge& operator=(const Ptr_StereoSGBM& ) { return *this; }
  Ptr_StereoSGBM toPtrStereoSGBM() { return Ptr_StereoSGBM(); }
  operator Ptr_StereoSGBM() { return toPtrStereoSGBM(); }

  // ---------------------------   Ptr_FeatureDetector   ----------------------
  Bridge& operator=(const Ptr_FeatureDetector& ) { return *this; }
  Ptr_FeatureDetector toPtrFeatureDetector() { return Ptr_FeatureDetector(); }
  operator Ptr_FeatureDetector() { return toPtrFeatureDetector(); }



};

#endif
