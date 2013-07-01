#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"
#include "map.hpp"
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

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


/*!
 * @brief raise error if condition fails
 *
 * This is a conditional wrapper for mexErrMsgTxt. If the conditional
 * expression fails, an error is raised and the mex function returns
 * to Matlab, otherwise this function does nothing
 */
void conditionalError(bool expr, const std::string& str) {
  if (!expr) mexErrMsgTxt(std::string("condition failed: ").append(str).c_str());
}

/*!
 * @brief raise an error
 *
 * This function is a wrapper around mexErrMsgTxt
 */
void error(const std::string& str) {
  mexErrMsgTxt(str.c_str());
}


// ----------------------------------------------------------------------------
//                          PREDECLARATIONS
// ----------------------------------------------------------------------------
class MxArray;
class Bridge;

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const cv::Mat& src, MxArray& dst);

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const MxArray& src, cv::Mat& dst);




// ----------------------------------------------------------------------------
//                            MATLAB TRAITS
// ----------------------------------------------------------------------------
namespace Matlab {
  class DefaultTraits {};
  class InheritType {};
  static const int Dynamic = -1;

  template<typename _Tp = DefaultTraits> class Traits {
  public:
    static const mxClassID ScalarType = mxUNKNOWN_CLASS;
    static const mxComplexity Complex = mxCOMPLEX;
    static const mxComplexity Real    = mxCOMPLEX;
    static std::string ToString()  { return "Unknown/Unsupported"; }
  };
  // bool
  template<> class Traits<bool> {
  public:
    static const mxClassID ScalarType = mxLOGICAL_CLASS;
    static std::string ToString()  { return "boolean"; }
  };
  // uint8_t
  template<> class Traits<uint8_t> {
  public:
    static const mxClassID ScalarType = mxUINT8_CLASS;
    static std::string ToString()  { return "uint8_t"; }
  };
  // int8_t
  template<> class Traits<int8_t> {
  public:
    static const mxClassID ScalarType = mxINT8_CLASS;
    static std::string ToString()  { return "int8_t"; }
  };
  // uint16_t
  template<> class Traits<uint16_t> {
  public:
    static const mxClassID ScalarType = mxUINT16_CLASS;
    static std::string ToString()  { return "uint16_t"; }
  };
  // int16_t
  template<> class Traits<int16_t> {
  public:
    static const mxClassID ScalarType = mxINT16_CLASS;
    static std::string ToString()  { return "int16_t"; }
  };
  // uint32_t
  template<> class Traits<uint32_t> {
  public:
    static const mxClassID ScalarType = mxUINT32_CLASS;
    static std::string ToString()  { return "uint32_t"; }
  };
  // int32_t
  template<> class Traits<int32_t> {
  public:
    static const mxClassID ScalarType = mxINT32_CLASS;
    static std::string ToString()  { return "int32_t"; }
  };
  // uint64_t
  template<> class Traits<uint64_t> {
  public:
    static const mxClassID ScalarType = mxUINT64_CLASS;
    static std::string ToString()  { return "uint64_t"; }
  };
  // int64_t
  template<> class Traits<int64_t> {
  public:
    static const mxClassID ScalarType = mxINT64_CLASS;
    static std::string ToString()  { return "int64_t"; }
  };
  // float
  template<> class Traits<float> {
  public:
    static const mxClassID ScalarType = mxSINGLE_CLASS;
    static std::string ToString()  { return "float"; }
  };
  // double
  template<> class Traits<double> {
  public:
    static const mxClassID ScalarType = mxDOUBLE_CLASS;
    static std::string ToString()  { return "double"; }
  };
  // size_t
  template<> class Traits<size_t> {
  public:
    static const mxClassID ScalarType = (sizeof(size_t) == 4) ? mxUINT32_CLASS : mxUINT64_CLASS;
    static std::string ToString()  { return "size_t"; }
  };
  // char
  template<> class Traits<char> {
  public:
    static const mxClassID ScalarType = mxCHAR_CLASS;
    static std::string ToString()  { return "char"; }
  };
  // char
  template<> class Traits<Matlab::InheritType> {
  public:
    static std::string ToString()  { return "Inherited type"; }
  };
}



// ----------------------------------------------------------------------------
//                                MXARRAY
// ----------------------------------------------------------------------------


/*!
 * @class MxArray
 * @brief A thin wrapper around Matlab's mxArray types
 *
 * MxArray provides a thin object oriented wrapper around Matlab's
 * native mxArray type which exposes most of the functionality of the
 * Matlab interface, but in a more C++ manner. MxArray objects are scoped,
 * so you can freely create and destroy them without worrying about memory
 * management. If you wish to pass the underlying mxArray* representation
 * back to Matlab as an lvalue, see the releaseOwnership() method
 */
class MxArray {
private:
  mxArray* ptr_;
  bool owns_;

  void dealloc() { 
    if (owns_ && ptr_) { mxDestroyArray(ptr_); ptr_ = NULL; owns_ = false; }
  }
public:
  // constructors and destructor
  MxArray() : ptr_(mxCreateDoubleMatrix(1, 1, Matlab::Traits<>::Real)), owns_(true) {}
  MxArray(const mxArray* ptr) : ptr_(const_cast<mxArray *>(ptr)), owns_(false) {}
  virtual ~MxArray() {
    dealloc();
  }
  // copy constructor
  // all copies are deep copies
  MxArray(const MxArray& other) : ptr_(mxDuplicateArray(other.ptr_)), owns_(true) {}
  // swap
  friend void swap(MxArray& first, MxArray& second) {
    using std::swap;
    swap(first.ptr_,  second.ptr_);
    swap(first.owns_, second.owns_);
  }
  // assignment operator
  // copy-and-swap idiom
  // all copies are deep copies
  MxArray& operator=(MxArray other) {
    swap(*this, other);
    return *this;
  }
#if __cplusplus >= 201103L
  // move constructor, if C++11
  // default construct and swap
  MxArray(MxArray&& other) : MxArray() {
    swap(*this, other);
  }
#endif

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


  template <typename Scalar>
  explicit MxArray(size_t m, size_t n, size_t k=1) : owns_(true) {
    mwSize dims[] = { static_cast<mwSize>(m), static_cast<mwSize>(n), static_cast<mwSize>(k) };
    ptr_ = mxCreateNumericArray(3, dims, Matlab::Traits<Scalar>::ScalarType, Matlab::Traits<>::Real);
  }

  // this function is called exclusively from constructors!!
  template <typename Scalar>
  MxArray& fromMat(const cv::Mat& mat) {
    // dealloc any existing storage before reallocating
    dealloc();
    mwSize dims[] = { static_cast<mwSize>(mat.rows), static_cast<mwSize>(mat.cols), static_cast<mwSize>(mat.channels()) };
    ptr_ = mxCreateNumericArray(3, dims, Matlab::Traits<Scalar>::ScalarType, Matlab::Traits<>::Real);
    owns_ = true;
    switch (mat.depth()) {
      case CV_8U:  deepCopyAndTranspose<uint8_t,  Scalar>(mat, *this); break;
      case CV_8S:  deepCopyAndTranspose<int8_t,   Scalar>(mat, *this); break;
      case CV_16U: deepCopyAndTranspose<uint16_t, Scalar>(mat, *this); break;
      case CV_16S: deepCopyAndTranspose<int16_t,  Scalar>(mat, *this); break;
      case CV_32S: deepCopyAndTranspose<int32_t,  Scalar>(mat, *this); break;
      case CV_32F: deepCopyAndTranspose<float,    Scalar>(mat, *this); break;
      case CV_64F: deepCopyAndTranspose<double,   Scalar>(mat, *this); break;
      default: error("Attempted to convert from unknown class");
    }
    return *this;
  }

  template <typename Scalar>
  cv::Mat toMat() const { 
    cv::Mat mat(cols(), rows(), CV_MAKETYPE(cv::DataType<Scalar>::type, channels()));
    switch (ID()) {
      case mxINT8_CLASS:    deepCopyAndTranspose<int8_t,   Scalar>(*this, mat); break;
      case mxUINT8_CLASS:   deepCopyAndTranspose<uint8_t,  Scalar>(*this, mat); break;
      case mxINT16_CLASS:   deepCopyAndTranspose<int16_t,  Scalar>(*this, mat); break;
      case mxUINT16_CLASS:  deepCopyAndTranspose<uint16_t, Scalar>(*this, mat); break;
      case mxINT32_CLASS:   deepCopyAndTranspose<int32_t,  Scalar>(*this, mat); break;
      case mxUINT32_CLASS:  deepCopyAndTranspose<uint32_t, Scalar>(*this, mat); break;
      case mxINT64_CLASS:   deepCopyAndTranspose<int64_t,  Scalar>(*this, mat); break;
      case mxUINT64_CLASS:  deepCopyAndTranspose<uint64_t, Scalar>(*this, mat); break;
      case mxSINGLE_CLASS:  deepCopyAndTranspose<float,    Scalar>(*this, mat); break;
      case mxDOUBLE_CLASS:  deepCopyAndTranspose<double,   Scalar>(*this, mat); break;
      case mxCHAR_CLASS:    deepCopyAndTranspose<char,     Scalar>(*this, mat); break;
      case mxLOGICAL_CLASS: deepCopyAndTranspose<int8_t,   Scalar>(*this, mat); break;
      default: error("Attempted to convert from unknown class");
    }
    return mat;
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


/*!
 * @brief template specialization for inheriting types
 *
 * This template specialization attempts to preserve the best mapping
 * between OpenCV and Matlab types. Matlab uses double types almost universally, so
 * all floating float types are converted to doubles.
 * Unfortunately OpenCV does not have a native logical type, so
 * that gets mapped to an unsigned 8-bit value
 */
template <>
MxArray& MxArray::fromMat<Matlab::InheritType>(const cv::Mat& mat) {
  switch (mat.depth()) {
    case CV_8U:  return fromMat<uint8_t>(mat);  break;
    case CV_8S:  return fromMat<int8_t>(mat);   break;
    case CV_16U: return fromMat<uint16_t>(mat); break;
    case CV_16S: return fromMat<int16_t>(mat);  break;
    case CV_32S: return fromMat<int32_t>(mat);  break;
    case CV_32F: return fromMat<double>(mat);   break; //NOTE: Matlab uses double as native type!
    case CV_64F: return fromMat<double>(mat);   break;
    default: error("Attempted to convert from unknown class");
  }
  return *this;
}

/*!
 * @brief template specialization for inheriting types
 * 
 * This template specialization attempts to preserve the best mapping
 * between Matlab and OpenCV types. OpenCV has poor support for double precision
 * types, so all floating point types are cast to float. Logicals get cast
 * to unsignd 8-bit value.
 */
template <>
cv::Mat MxArray::toMat<Matlab::InheritType>() const {
  switch (ID()) {
    case mxINT8_CLASS:    return toMat<int8_t>();
    case mxUINT8_CLASS:   return toMat<uint8_t>();;
    case mxINT16_CLASS:   return toMat<int16_t>();
    case mxUINT16_CLASS:  return toMat<uint16_t>();
    case mxINT32_CLASS:   return toMat<int32_t>();
    case mxUINT32_CLASS:  return toMat<int32_t>();
    case mxINT64_CLASS:   return toMat<int64_t>();
    case mxUINT64_CLASS:  return toMat<int64_t>();
    case mxSINGLE_CLASS:  return toMat<float>();
    case mxDOUBLE_CLASS:  return toMat<float>(); //NOTE: OpenCV uses float as native type!
    case mxCHAR_CLASS:    return toMat<int8_t>();
    case mxLOGICAL_CLASS: return toMat<int8_t>();
    default: error("Attempted to convert from unknown class");
  }
  return cv::Mat();
}



// ----------------------------------------------------------------------------
//                            MATRIX TRANSPOSE
// ----------------------------------------------------------------------------

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const cv::Mat& in, MxArray& out) {
  conditionalError(static_cast<size_t>(in.rows) == out.rows(), "Matrices must have the same number of rows");
  conditionalError(static_cast<size_t>(in.cols) == out.cols(), "Matrices must have the same number of cols");
  conditionalError(static_cast<size_t>(in.channels()) == out.channels(), "Matrices must have the same number of channels");
  OutputScalar* outp = out.real<OutputScalar>();
  const size_t M = out.rows();
  const size_t N = out.cols();
  for (size_t m = 0; m < M; ++m) {
    const InputScalar* inp = in.ptr<InputScalar>(m);
    for (size_t n = 0; n < N; ++n) {
      // copy and transpose
      outp[m + n*M] = inp[n];
    }
  }
}

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const MxArray& in, cv::Mat& out) {
  conditionalError(in.rows() == static_cast<size_t>(out.rows), "Matrices must have the same number of rows");
  conditionalError(in.cols() == static_cast<size_t>(out.cols), "Matrices must have the same number of cols");
  conditionalError(in.channels() == static_cast<size_t>(out.channels()), "Matrices must have the same number of channels");
  const InputScalar* inp = in.real<InputScalar>();
  const size_t M = in.rows();
  const size_t N = in.cols();
  for (size_t m = 0; m < M; ++m) {
    OutputScalar* outp = out.ptr<OutputScalar>(m);
    for (size_t n = 0; n < N; ++n) {
      // copy and transpose
      outp[n] = inp[m + n*M];
    }
  }
}


template <> 
void deepCopyAndTranspose<float, float>(const cv::Mat&, MxArray&) {
}

template <> 
void deepCopyAndTranspose<double, double>(const cv::Mat&, MxArray&) {
}

template <> 
void deepCopyAndTranspose<float, float>(const MxArray&, cv::Mat&) {
  // use mkl
}

template <> 
void deepCopyAndTranspose<double, double>(const MxArray&, cv::Mat& ) {
  // use mkl
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
  Bridge& operator=(const mxArray*) { return *this; }
  Bridge(const mxArray* obj) : ptr_(obj) {}
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
  Bridge& operator=(const cv::Mat& mat) { ptr_ = MxArray().fromMat<Matlab::InheritType>(mat); return *this; }
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
