#ifndef OPENCV_MXARRAY_HPP_
#define OPENCV_MXARRAY_HPP_

#include "mex.h"
#include "transpose.hpp"
#include <vector>
#include <string>
#include <opencv2/core.hpp>

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
#ifdef __cplusplus
}
#endif


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
 *
 * MxArrays can be directly converted into OpenCV mat objects and std::string
 * objects, since there is a natural mapping between these types. More
 * complex types are mapped through the Bridge which does custom conversions
 * such as MxArray --> cv::Keypoints, etc
 */
class MxArray {
private:
  mxArray* ptr_;
  bool owns_;

  /*!
   * @brief swap all members of this and other
   *
   * the swap method is used by the assignment and move constructors 
   * to swap the members of two MxArrays, leaving both in destructible states
   */
  friend void swap(MxArray& first, MxArray& second) {
    using std::swap;
    swap(first.ptr_,  second.ptr_);
    swap(first.owns_, second.owns_);
  }

  void dealloc() { 
    if (owns_ && ptr_) { mxDestroyArray(ptr_); ptr_ = NULL; owns_ = false; }
  }
public:
  // --------------------------------------------------------------------------
  //                              CONSTRUCTORS
  // --------------------------------------------------------------------------
  /*!
   * @brief default constructor
   *
   * Construct a valid 0x0 matrix (so all other methods do not need validity checks
   */
  MxArray() : ptr_(mxCreateDoubleMatrix(1, 1, Matlab::Traits<>::Real)), owns_(true) {}

  /*!
   * @brief inheriting constructor
   *
   * Inherit an mxArray from Matlab. Don't claim ownership of the array,
   * just encapsulate it
   */
  MxArray(const mxArray* ptr) : ptr_(const_cast<mxArray *>(ptr)), owns_(false) {}
  MxArray& operator=(const mxArray* ptr) { 
    dealloc();
    ptr_ = const_cast<mxArray *>(ptr);
    owns_ = false;
    return *this;
  }

  /*!
   * @brief explicit typed constructor
   *
   * This constructor explicitly creates an MxArray of the given size and type.
   */
  MxArray(size_t m, size_t n, size_t k, mxClassID id, mxComplexity com = Matlab::Traits<>::Real) : owns_(true) {
    mwSize dims[] = { static_cast<mwSize>(m), static_cast<mwSize>(n), static_cast<mwSize>(k) };
    ptr_ = mxCreateNumericArray(3, dims, id, com);
  }

  /*!
   * @brief explicit tensor constructor
   *
   * Explicitly construct a tensor of given size and type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Tensor(size_t m, size_t n, size_t k=1) {
    return MxArray(m, n, k, Matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit matrix constructor
   *
   * Explicitly construct a matrix of given size and type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Matrix(size_t m, size_t n) {
    return MxArray(m, n, 1, Matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit vector constructor
   *
   * Explicitly construct a vector of given size and type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Vector(size_t m) {
    return MxArray(m, 1, 1, Matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit scalar constructor
   *
   * Explicitly construct a scalar of given type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Scalar(Scalar value = 0) {
    MxArray s(1, 1, 1, Matlab::Traits<Scalar>::ScalarType);
    s.real<Scalar>()[0] = value;
    return s;
  }

  /*! 
   * @brief destructor
   * 
   * The destructor deallocates any data allocated by mxCreate* methods only
   * if the object is owned
   */
  virtual ~MxArray() {
    dealloc();
  }

  /*! 
   * @brief copy constructor
   *
   * All copies are deep copies. If you have a C++11 compatible compiler, prefer
   * move construction to copy construction
   */
  MxArray(const MxArray& other) : ptr_(mxDuplicateArray(other.ptr_)), owns_(true) {}

  /*!
   * @brief copy-and-swap assignment
   *
   * This assignment operator uses the copy and swap idiom to provide a strong
   * exception guarantee when swapping two objects. 
   *
   * Note in particular that the other MxArray is passed by value, thus invoking
   * the copy constructor which performs a deep copy of the input. The members of
   * this and other are then swapped
   */
  MxArray& operator=(MxArray other) {
    swap(*this, other);
    return *this;
  }
#if __cplusplus >= 201103L
  /*
   * @brief C++11 move constructor
   *
   * When C++11 support is available, move construction is used to move returns
   * out of functions, etc. This is much fast than copy construction, since the
   * move constructed object replaced itself with a default constructed MxArray,
   * which is of size 0 x 0.
   */
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
  static MxArray FromMat(const cv::Mat& mat) {
    MxArray arr(mat.rows, mat.cols, mat.channels(), Matlab::Traits<Scalar>::ScalarType);
    switch (mat.depth()) {
      case CV_8U:  deepCopyAndTranspose<uint8_t,  Scalar>(mat, arr); break;
      case CV_8S:  deepCopyAndTranspose<int8_t,   Scalar>(mat, arr); break;
      case CV_16U: deepCopyAndTranspose<uint16_t, Scalar>(mat, arr); break;
      case CV_16S: deepCopyAndTranspose<int16_t,  Scalar>(mat, arr); break;
      case CV_32S: deepCopyAndTranspose<int32_t,  Scalar>(mat, arr); break;
      case CV_32F: deepCopyAndTranspose<float,    Scalar>(mat, arr); break;
      case CV_64F: deepCopyAndTranspose<double,   Scalar>(mat, arr); break;
      default: error("Attempted to convert from unknown class");
    }
    return arr;
  }

  template <typename Scalar>
  cv::Mat toMat() const { 
    cv::Mat mat(rows(), cols(), CV_MAKETYPE(cv::DataType<Scalar>::type, channels()));
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
  Scalar* imag() { return static_cast<Scalar *>(mxGetImagData(ptr_)); }

  template <typename Scalar>
  const Scalar* real() const { return static_cast<const Scalar *>(mxGetData(ptr_)); }
  
  template <typename Scalar>
  const Scalar* imag() const { return static_cast<const Scalar *>(mxGetData(ptr_)); }

  template <typename Scalar>
  Scalar scalar() const { return static_cast<double *>(mxGetData(ptr_))[0]; }

  std::string toString() const {
    conditionalError(isString(), "Attempted to convert non-string type to string");
    std::string str(size()+1, '\0');
    mxGetString(ptr_, const_cast<char *>(str.data()), str.size());
    mexPrintf("string: %s\n", str.c_str());
    return str;
  }

  size_t size() const { return mxGetNumberOfElements(ptr_); }
  size_t rows() const { return mxGetDimensions(ptr_)[0]; }
  size_t cols() const { return mxGetDimensions(ptr_)[1]; }
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
MxArray MxArray::FromMat<Matlab::InheritType>(const cv::Mat& mat) {
  switch (mat.depth()) {
    case CV_8U:  return FromMat<uint8_t>(mat);
    case CV_8S:  return FromMat<int8_t>(mat);
    case CV_16U: return FromMat<uint16_t>(mat);
    case CV_16S: return FromMat<int16_t>(mat);
    case CV_32S: return FromMat<int32_t>(mat);
    case CV_32F: return FromMat<double>(mat); //NOTE: Matlab uses double as native type!
    case CV_64F: return FromMat<double>(mat);
    default: error("Attempted to convert from unknown class");
  }
  return MxArray();
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
    case mxUINT8_CLASS:   return toMat<uint8_t>();
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
  std::vector<cv::Mat> channels;
  cv::split(in, channels);
  for (size_t c = 0; c < out.channels(); ++c) {
    cv::transpose(channels[c], channels[c]);
    cv::Mat outmat(out.cols(), out.rows(), cv::DataType<OutputScalar>::type, 
      static_cast<void *>(out.real<OutputScalar>() + out.cols()*out.rows()*c));
    channels[c].convertTo(outmat, cv::DataType<OutputScalar>::type);
  }

  //const InputScalar* inp = in.ptr<InputScalar>(0);
  //OutputScalar* outp = out.real<OutputScalar>();
  //gemt('R', out.rows(), out.cols(), inp, in.step1(), outp, out.rows());
}

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const MxArray& in, cv::Mat& out) {
  conditionalError(in.rows() == static_cast<size_t>(out.rows), "Matrices must have the same number of rows");
  conditionalError(in.cols() == static_cast<size_t>(out.cols), "Matrices must have the same number of cols");
  conditionalError(in.channels() == static_cast<size_t>(out.channels()), "Matrices must have the same number of channels");
  std::vector<cv::Mat> channels;
  for (size_t c = 0; c < in.channels(); ++c) {
    cv::Mat outmat;
    cv::Mat inmat(in.cols(), in.rows(), cv::DataType<InputScalar>::type,
      static_cast<void *>(const_cast<InputScalar *>(in.real<InputScalar>() + in.cols()*in.rows()*c)));
    inmat.convertTo(outmat, cv::DataType<OutputScalar>::type);
    cv::transpose(outmat, outmat);
    channels.push_back(outmat);
  }
  cv::merge(channels, out);

  //const InputScalar* inp = in.real<InputScalar>();
  //OutputScalar* outp = out.ptr<OutputScalar>(0);
  //gemt('C', in.rows(), in.cols(), inp, in.rows(), outp, out.step1()); 
}



#endif
