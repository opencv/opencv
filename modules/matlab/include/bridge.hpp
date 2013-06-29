#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <ext/hash_map>

/*
 * All recent versions of Matlab ship with the MKL library which contains
 * a blas extension called mkl_?omatcopy(). This  defines an out-of-place 
 * copy and transpose operation.
 *
 * The mkl library is in ${MATLAB_ROOT}/bin/${MATLAB_MEXEXT}/libmkl...
 * Matlab does not ship headers for the mkl functions, so we define them
 * here.
 *
 * This operation is used extensively to copy between Matlab's column-major
 * format and OpenCV's row-major format.
 */
#ifdef __cplusplus
extern "C" {
#endif
  void mkl_somatcopy(char, char, size_t, size_t, const float,  const float*,  size_t, float*,  size_t);
  void mkl_domatcopy(char, char, size_t, size_t, const double, const double*, size_t, double*, size_t);
#ifdef __cplusplus
}
#endif

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


void conditionalError(bool expr, const std::string& str) {
  if (!expr) mexErrMsgTxt(std::string("condition failed: ").append(str).c_str());
}

void error(const std::string& str) {
  mexErrMsgTxt(str.c_str());
}


namespace Matlab {
  class DefaultTraits {};
  class InheritType {};
  static const int Dynamic = -1;

  // --------------------------------------------------------------------------
  //                       INTERNAL TRAITS CLASS
  // --------------------------------------------------------------------------
  template<typename _Tp = DefaultTraits> class Traits {
  public:
    static const mxClassID ScalarType = mxUNKNOWN_CLASS;
    static const mxComplexity Complex = mxCOMPLEX;
    static const mxComplexity Real    = mxCOMPLEX;
  };
  // bool
  template<> class Traits<bool> {
  public:
    static const mxClassID ScalarType = mxLOGICAL_CLASS;
  };
  // uint8_t
  template<> class Traits<uint8_t> {
  public:
    static const mxClassID ScalarType = mxUINT8_CLASS;
  };
  // int8_t
  template<> class Traits<int8_t> {
  public:
    static const mxClassID ScalarType = mxINT8_CLASS;
  };
  // uint16_t
  template<> class Traits<uint16_t> {
  public:
    static const mxClassID ScalarType = mxUINT16_CLASS;
  };
  // int16_t
  template<> class Traits<int16_t> {
  public:
    static const mxClassID ScalarType = mxINT16_CLASS;
  };
  // uint32_t
  template<> class Traits<uint32_t> {
  public:
    static const mxClassID ScalarType = mxUINT32_CLASS;
  };
  // int32_t
  template<> class Traits<int32_t> {
  public:
    static const mxClassID ScalarType = mxINT32_CLASS;
  };
  // uint64_t
  template<> class Traits<uint64_t> {
  public:
    static const mxClassID ScalarType = mxUINT64_CLASS;
  };
  // int64_t
  template<> class Traits<int64_t> {
  public:
    static const mxClassID ScalarType = mxINT64_CLASS;
  };
  // float
  template<> class Traits<float> {
  public:
    static const mxClassID ScalarType = mxSINGLE_CLASS;
  };
  // double
  template<> class Traits<double> {
  public:
    static const mxClassID ScalarType = mxDOUBLE_CLASS;
  };
  // size_t
  template<> class Traits<size_t> {
  public:
    static const mxClassID ScalarType = (sizeof(size_t) == 4) ? mxUINT32_CLASS : mxUINT64_CLASS;
  };
  // char
  template<> class Traits<char> {
  public:
    static const mxClassID ScalarType = mxCHAR_CLASS;
  };
};



// ----------------------------------------------------------------------------
//                                MXARRAY
// ----------------------------------------------------------------------------


/*!
 * @class MxArray
 * @brief A thin wrapper around Matlab's mxArray types
 *
 * MxArray provides a thin object oriented wrapper around Matlab's
 * native mxArray type which exposes most of the functionality of the
 * Matlab interface, but in a more C++ manner.
 */
class MxArray {
private:
  mxArray* ptr_;
  bool owns_;
public:
  MxArray() : ptr_(NULL), owns_(false) {}
  MxArray(const mxArray* ptr) : ptr_(const_cast<mxArray *>(ptr)), owns_(false) {}
  ~MxArray() {
    if (owns_ && ptr_) mxDestroyArray(ptr_);
  }

  /*
   * @brief release ownership to allow return into Matlab workspace
   *
   * MxArray is not directly convertible back to mxArray types through assignment
   * because the MxArray may have been allocated on the free store, making it impossible
   * to know whether the returned pointer will be released by someone else or not.
   * 
   * Since Matlab requires mxArrays be passed back into the workspace, the only way
   * to achieve that is through this function, which explicitly releases ownership 
   * of the object, assuming the Matlab interpreter receving the object will delete
   * it at a later time
   *
   * e.g.
   * {
   *    MxArray A<double>(5, 5);        // allocates memory
   *    MxArray B<double>(5, 5);        // ditto
   *    plhs[0] = A;                    // not allowed!!
   *    plhs[0] = A.releaseOwnership(); // makes explicit that ownership is being released
   * } // end of scope. B is released, A isn't
   *
   */
  mxArray* releaseOwnership() {
    owns_ = false;
    return ptr_;
  }


  // TODO: Make sure NULL pointers are checked in all functions!
  template <typename Scalar>
  explicit MxArray(size_t m, size_t n, size_t k=1) : owns_(true) {
    mwSize dims[] = {m, n, k};
    ptr_ = mxCreateNumericArray(3, dims, Matlab::Traits<Scalar>::ScalarType, Matlab::Traits<>::Real);
  }

  template <typename Scalar>
  explicit MxArray(const cv::Mat& mat) : owns_(true) {
    mwSize dims[] = { mat.rows, mat.cols, mat.channels() };
    if (mat.channels() == 2) {
      ptr_ = mxCreateNumericArray(2, dims, Matlab::Traits<Scalar>::ScalarType, Matlab::Traits<>::Complex);
    } else {
      ptr_ = mxCreateNumericArray(2, dims, Matlab::Traits<Scalar>::ScalarType, Matlab::Traits<>::Real);
    }
  }

  template <typename Scalar>
  cv::Mat toMat() const { 
    cv::Mat mat(cols(), rows(), CV_MAKETYPE(cv::DataType<Scalar>::type, channels()));
    return mat;
  }

  template <typename Scalar>
  void fromMat(const cv::Mat& mat) {

  }

  MxArray field(const std::string& name) { return MxArray(mxGetField(ptr_, 0, name.c_str())); }

  template <typename Scalar>
  Scalar* real() { return static_cast<Scalar *>(mxGetData(ptr_)); }
  
  template <typename Scalar>
  Scalar* imag() { return static_cast<Scalar *>(mxGetData(ptr_)); }

  template <typename Scalar>
  const Scalar* real() const { return static_cast<const Scalar *>(mxGetData(ptr_)); }
  
  template <typename Scalar>
  const Scalar* imag() const { return static_cast<const Scalar *>(mxGetData(ptr_)); }

  template <typename Scalar>
  Scalar scalar() const { return static_cast<Scalar *>(mxGetData(ptr_))[0]; }

  std::string toString() const {
    conditionalError(isString(), "Attempted to convert non-string type to string");
    std::string str;
    str.reserve(size()+1);
    mxGetString(ptr_, const_cast<char *>(str.data()), str.size());
    return str;
  }

  size_t size() const { return mxGetNumberOfElements(ptr_); }
  size_t rows() const { return mxGetM(ptr_); }
  size_t cols() const { return mxGetN(ptr_); }
  size_t channels() const { return (mxGetNumberOfDimensions(ptr_) > 2) ? mxGetDimensions(ptr_)[2] : 1; }
  bool isComplex() const { return mxIsComplex(ptr_); }
  bool isNumeric() const { return mxIsNumeric(ptr_); }
  bool isLogical() const { return mxIsLogical(ptr_); }
  bool isString() const { return mxIsChar(ptr_); }
  bool isCell() const { return mxIsCell(ptr_); }
  bool isStructure() const { return mxIsStruct(ptr_); }
  bool isClass(const std::string& name) const { return mxIsClass(ptr_, name.c_str()); }
  std::string className() const { return std::string(mxGetClassName(ptr_)); }
  mxClassID ID() const { return mxGetClassID(ptr_); }

};


template <>
cv::Mat MxArray::toMat<Matlab::InheritType>() const {
  return cv::Mat();
}

template <>
void MxArray::fromMat<Matlab::InheritType>(const cv::Mat& mat) {

}



// ----------------------------------------------------------------------------
//                            MATRIX TRANSPOSE
// ----------------------------------------------------------------------------

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const cv::Mat& src, MxArray& dst) {
}

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const MxArray& src, cv::Mat& dst) {
}

template <> 
void deepCopyAndTranspose<float, float>(const cv::Mat& src, MxArray& dst) {
}

template <> 
void deepCopyAndTranspose<double, double>(const cv::Mat& src, MxArray& dst) {
}

template <> 
void deepCopyAndTranspose<float, float>(const MxArray& src, cv::Mat& dst) {
}

template <> 
void deepCopyAndTranspose<double, double>(const MxArray& src, cv::Mat& dst) {
}



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
  Bridge& operator=(const mxArray* obj) { return *this; }
  Bridge(const mxArray* obj) : ptr_(obj) {}
  MxArray toMxArray() { return ptr_; }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                         INTEGRAL TYPES
  // --------------------------------------------------------------------------
  
  // --------------------------- string  --------------------------------------
  Bridge& operator=(const std::string& obj) { return *this; }
  std::string toString() { 
    return ptr_.toString();
  }
  operator std::string() { return toString(); }

  // ---------------------------  bool   --------------------------------------
  Bridge& operator=(const bool& obj) { return *this; }
  bool toBool() { return 0; }
  operator bool() { return toBool(); }

  // --------------------------- double  --------------------------------------
  Bridge& operator=(const double& obj) { return *this; }
  double toDouble() { return ptr_.scalar<double>(); }
  operator double() { return toDouble(); }

  // --------------------------- float  ---------------------------------------
  Bridge& operator=(const float& obj) { return *this; }
  float toFloat() { return ptr_.scalar<float>(); }
  operator float() { return toFloat(); }

  // ---------------------------   int   --------------------------------------
  Bridge& operator=(const int& obj) { return *this; }
  int toInt() { return ptr_.scalar<int>(); }
  operator int() { return toInt(); }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                       CORE OPENCV TYPES
  // --------------------------------------------------------------------------

  // --------------------------- cv::Mat --------------------------------------
  Bridge& operator=(const cv::Mat& obj) { return *this; }
  cv::Mat toMat() const { return ptr_.toMat<Matlab::InheritType>(); }
  operator cv::Mat() const { return toMat(); }
  
  // --------------------------   Point  --------------------------------------
  Bridge& operator=(const cv::Point& obj) { return *this; }
  cv::Point toPoint() const { return cv::Point(); }
  operator cv::Point() const { return toPoint(); }
  
  // --------------------------   Point2f  ------------------------------------
  Bridge& operator=(const cv::Point2f& obj) { return *this; }
  cv::Point2f toPoint2f() const { return cv::Point2f(); }
  operator cv::Point2f() const { return toPoint2f(); }
  
  // --------------------------   Point2d  ------------------------------------
  Bridge& operator=(const cv::Point2d& obj) { return *this; }
  cv::Point2d toPoint2d() const { return cv::Point2d(); }
  operator cv::Point2d() const { return toPoint2d(); }
  
  // --------------------------   Size  ---------------------------------------
  Bridge& operator=(const cv::Size& obj) { return *this; }
  cv::Size toSize() const { return cv::Size(); }
  operator cv::Size() const { return toSize(); }
  
  // -------------------------- Moments  --------------------------------------
  Bridge& operator=(const cv::Moments& obj) { return *this; }
  cv::Moments toMoments() const { return cv::Moments(); }
  operator cv::Moments() const { return toMoments(); }
  
  // --------------------------  Scalar  --------------------------------------
  Bridge& operator=(const cv::Scalar& obj) { return *this; }
  cv::Scalar toScalar() { return cv::Scalar(); }
  operator cv::Scalar() { return toScalar(); }
  
  // -------------------------- Rect  -----------------------------------------
  Bridge& operator=(const cv::Rect& obj) { return *this; }
  cv::Rect toRect() { return cv::Rect(); }
  operator cv::Rect() { return toRect(); }
  
  // ---------------------- RotatedRect ---------------------------------------
  Bridge& operator=(const cv::RotatedRect& obj) { return *this; }
  cv::RotatedRect toRotatedRect() { return cv::RotatedRect(); }
  operator cv::RotatedRect() { return toRotatedRect(); }
  
  // ---------------------- TermCriteria --------------------------------------
  Bridge& operator=(const cv::TermCriteria& obj) { return *this; }
  cv::TermCriteria toTermCriteria() { return cv::TermCriteria(); }
  operator cv::TermCriteria() { return toTermCriteria(); }
  
  // ----------------------      RNG     --------------------------------------
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

  
  
  

  // --------------------------------------------------------------------------
  //                       OPENCV VECTOR TYPES
  // --------------------------------------------------------------------------
  
  // -------------------- vector_Mat ------------------------------------------
  Bridge& operator=(const vector_Mat& obj) { return *this; }
  vector_Mat toVectorMat() { return vector_Mat(); }
  operator vector_Mat() { return toVectorMat(); }

  // --------------------------- vector_int  ----------------------------------
  Bridge& operator=(const vector_int& obj) { return *this; }
  vector_int toVectorInt() { return vector_int(); }
  operator vector_int() { return toVectorInt(); }
  
  // --------------------------- vector_float  --------------------------------
  Bridge& operator=(const vector_float& obj) { return *this; }
  vector_float toVectorFloat() { return vector_float(); }
  operator vector_float() { return toVectorFloat(); }
  
  // --------------------------- vector_Rect  ---------------------------------
  Bridge& operator=(const vector_Rect& obj) { return *this; }
  vector_Rect toVectorRect() { return vector_Rect(); }
  operator vector_Rect() { return toVectorRect(); }
  
  // --------------------------- vector_KeyPoint  -----------------------------
  Bridge& operator=(const vector_KeyPoint& obj) { return *this; }
  vector_KeyPoint toVectorKeyPoint() { return vector_KeyPoint(); }
  operator vector_KeyPoint() { return toVectorKeyPoint(); }
  
  // --------------------------- vector_String  -------------------------------
  Bridge& operator=(const vector_String& obj) { return *this; }
  vector_String toVectorString() { return vector_String(); }
  operator vector_String() { return toVectorString(); }
  
  // ------------------------ vector_Point ------------------------------------
  Bridge& operator=(const vector_Point& obj) { return *this; }
  vector_Point toVectorPoint() { return vector_Point(); }
  operator vector_Point() { return toVectorPoint(); }
  
  // ------------------------ vector_uchar ------------------------------------
  Bridge& operator=(const vector_uchar& obj) { return *this; }
  vector_uchar toVectorUchar() { return vector_uchar(); }
  operator vector_uchar() { return toVectorUchar(); }
  
  
  
  
  
  // --------------------------------------------------------------------------
  //                       OPENCV COMPLEX TYPES
  // --------------------------------------------------------------------------

  // ---------------------------   Ptr_StereoBM   -----------------------------
  Bridge& operator=(const Ptr_StereoBM& obj) { return *this; }
  Ptr_StereoBM toPtrStereoBM() { return Ptr_StereoBM(); }
  operator Ptr_StereoBM() { return toPtrStereoBM(); }

  // ---------------------------   Ptr_StereoSGBM   ---------------------------
  Bridge& operator=(const Ptr_StereoSGBM& obj) { return *this; }
  Ptr_StereoSGBM toPtrStereoSGBM() { return Ptr_StereoSGBM(); }
  operator Ptr_StereoSGBM() { return toPtrStereoSGBM(); }

  // ---------------------------   Ptr_FeatureDetector   ----------------------
  Bridge& operator=(const Ptr_FeatureDetector& obj) { return *this; }
  Ptr_FeatureDetector toPtrFeatureDetector() { return Ptr_FeatureDetector(); }
  operator Ptr_FeatureDetector() { return toPtrFeatureDetector(); }



};

#endif
