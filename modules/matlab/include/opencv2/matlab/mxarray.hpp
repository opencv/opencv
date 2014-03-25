////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef OPENCV_MXARRAY_HPP_
#define OPENCV_MXARRAY_HPP_

#include <mex.h>
#include <stdint.h>
#include <cstdarg>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#if __cplusplus > 201103
#include <unordered_set>
typedef std::unordered_set<std::string> StringSet;
#else
#include <set>
typedef std::set<std::string> StringSet;
#endif

/*
 * All recent versions of Matlab ship with the MKL library which contains
 * a blas extension called mkl_?omatcopy(). This  defines an out-of-place
 * copy and transpose operation.
 *
 * The mkl library is in ${MATLAB_ROOT}/bin/${MATLAB_MEXEXT}/libmkl...
 * Matlab does not ship headers for the mkl functions, so we define them
 * here.
 *
 */
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

namespace matlab {
// ----------------------------------------------------------------------------
//                          PREDECLARATIONS
// ----------------------------------------------------------------------------
class MxArray;
typedef std::vector<MxArray> MxArrayVector;

/*!
 * @brief raise error if condition fails
 *
 * This is a conditional wrapper for mexErrMsgTxt. If the conditional
 * expression fails, an error is raised and the mex function returns
 * to Matlab, otherwise this function does nothing
 */
static void conditionalError(bool expr, const std::string& str) {
  if (!expr) mexErrMsgTxt(std::string("condition failed: ").append(str).c_str());
}

/*!
 * @brief raise an error
 *
 * This function is a wrapper around mexErrMsgTxt
 */
static void error(const std::string& str) {
  mexErrMsgTxt(str.c_str());
}


// ----------------------------------------------------------------------------
//                            MATLAB TRAITS
// ----------------------------------------------------------------------------
class DefaultTraits {};
class InheritType {};

template<typename _Tp = DefaultTraits> class Traits {
public:
  static const mxClassID ScalarType = mxUNKNOWN_CLASS;
  static const mxComplexity Complex = mxCOMPLEX;
  static const mxComplexity Real    = mxREAL;
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
// char
template<> class Traits<char> {
public:
  static const mxClassID ScalarType = mxCHAR_CLASS;
  static std::string ToString()  { return "char"; }
};
// inherited type
template<> class Traits<matlab::InheritType> {
public:
  static std::string ToString()  { return "Inherited type"; }
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
   * Construct a valid 0x0 matrix (so all other methods do not need validity checks)
   */
  MxArray() : ptr_(mxCreateDoubleMatrix(0, 0, matlab::Traits<>::Real)), owns_(true) {}

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
  MxArray(size_t m, size_t n, size_t k, mxClassID id, mxComplexity com = matlab::Traits<>::Real)
      : ptr_(NULL), owns_(true) {
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
    return MxArray(m, n, k, matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit matrix constructor
   *
   * Explicitly construct a matrix of given size and type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Matrix(size_t m, size_t n) {
    return MxArray(m, n, 1, matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit vector constructor
   *
   * Explicitly construct a vector of given size and type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename Scalar>
  static MxArray Vector(size_t m) {
    return MxArray(m, 1, 1, matlab::Traits<Scalar>::ScalarType);
  }

  /*!
   * @brief explicit scalar constructor
   *
   * Explicitly construct a scalar of given type. Since constructors cannot
   * be explicitly templated, this is a static factory method
   */
  template <typename ScalarType>
  static MxArray Scalar(ScalarType value = 0) {
    MxArray s(1, 1, 1, matlab::Traits<ScalarType>::ScalarType);
    s.real<ScalarType>()[0] = value;
    return s;
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
   *    MxArray A = MxArray::Matrix<double>(5, 5);  // allocates memory
   *    MxArray B = MxArray::Matrix<double>(5, 5);  // ditto
   *    plhs[0] = A;                                // not allowed!!
   *    plhs[0] = A.releaseOwnership();             // makes explicit that ownership is being released
   * } // end of scope. B is released, A isn't
   *
   */
  mxArray* releaseOwnership() {
    owns_ = false;
    return ptr_;
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
  Scalar scalar() const { return static_cast<Scalar *>(mxGetData(ptr_))[0]; }

  std::string toString() const {
    conditionalError(isString(), "Attempted to convert non-string type to string");
    std::string str(size(), '\0');
    mxGetString(ptr_, const_cast<char *>(str.data()), str.size()+1);
    return str;
  }

  size_t size() const { return mxGetNumberOfElements(ptr_); }
  bool empty() const { return size() == 0; }
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


// ----------------------------------------------------------------------------
//                           ARGUMENT PARSER
// ----------------------------------------------------------------------------

/*! @class ArgumentParser
 *  @brief parses inputs to a method and resolves the argument names.
 *
 * The ArgumentParser resolves the inputs to a method. It checks that all
 * required arguments are specified and also allows named optional arguments.
 * For example, the C++ function:
 *    void randn(Mat& mat, Mat& mean=Mat(), Mat& std=Mat());
 * could be called in Matlab using any of the following signatures:
 * \code
 *    out = randn(in);
 *    out = randn(in, 0, 1);
 *    out = randn(in, 'mean', 0, 'std', 1);
 * \endcode
 *
 * ArgumentParser also enables function overloading by allowing users
 * to add variants to a method. For example, there may be two C++ sum() methods:
 * \code
 *    double sum(Mat& mat);     % sum elements of a matrix
 *    Mat sum(Mat& A, Mat& B);  % add two matrices
 * \endcode
 *
 * by adding two variants to ArgumentParser, the correct underlying sum
 * method can be called. If the function call is ambiguous, the
 * ArgumentParser will fail with an error message.
 *
 * The previous example could be parsed as:
 * \code
 *    // set up the Argument parser
 *    ArgumentParser arguments;
 *    arguments.addVariant("elementwise", 1);
 *    arguments.addVariant("matrix", 2);
 *
 *    // parse the arguments
 *    std::vector<MxArray> inputs;
 *    inputs = arguments.parse(std::vector<MxArray>(prhs, prhs+nrhs));
 *
 *    // if we get here, one unique variant is valid
 *    if (arguments.variantIs("elementwise")) {
 *      // call elementwise sum()
 *    }
 * \endcode
 */
class ArgumentParser {
private:
  struct Variant;
  typedef std::string String;
  typedef std::vector<std::string> StringVector;
  typedef std::vector<size_t> IndexVector;
  typedef std::vector<Variant> VariantVector;

  /* @class Variant
   * @brief Describes a variant of arguments to a method
   *
   * When addVariant() is called on an instance to ArgumentParser, this class
   * holds the the information that decribes that variant. The parse() method
   * of ArgumentParser then attempts to match a Variant, given a set of
   * inputs for a method invocation.
   */
  class Variant {
  private:
    String name_;
    size_t Nreq_;
    size_t Nopt_;
    StringVector keys_;
    IndexVector order_;
    bool valid_;
    size_t nparsed_;
    size_t nkeys_;
    size_t working_opt_;
    bool expecting_val_;
    bool using_named_;
    size_t find(const String& key) const {
      return std::find(keys_.begin(), keys_.end(), key) - keys_.begin();
    }
  public:
    /*! @brief default constructor */
    Variant() : Nreq_(0), Nopt_(0), valid_(false) {}
    /*! @brief construct a new variant spec */
    Variant(const String& name, size_t Nreq, size_t Nopt, const StringVector& keys)
      : name_(name), Nreq_(Nreq), Nopt_(Nopt), keys_(keys),
      order_(Nreq+Nopt, Nreq+2*Nopt), valid_(true), nparsed_(0), nkeys_(0),
      working_opt_(0), expecting_val_(false), using_named_(false) {}
    /*! @brief the name of the variant */
    String name() const { return name_; }
    /*! @brief return the total number of arguments the variant can take */
    size_t size() const { return Nreq_ + Nopt_; }
    /*! @brief has the variant been fulfilled? */
    bool fulfilled() const { return (valid_ && nparsed_ >= Nreq_ && !expecting_val_); }
    /*! @brief is the variant in a valid state (though not necessarily fulfilled) */
    bool valid() const { return valid_; }
    /*! @brief check if the named argument exists in the variant */
    bool exist(const String& key) const { return find(key) != keys_.size(); }
    /*! @brief retrieve the order mapping raw inputs to their position in the variant */
    const IndexVector& order() const { return order_; }
    size_t order(size_t n) const { return order_[n]; }
    /*! @brief attempt to parse the next argument as a value */
    bool parseNextAsValue() {
      if (!valid_) {}
      else if ((using_named_ && !expecting_val_) || (nparsed_-nkeys_ == Nreq_+Nopt_)) { valid_ = false; }
      else if (nparsed_ < Nreq_) { order_[nparsed_] = nparsed_; }
      else if (!using_named_) { order_[nparsed_] = nparsed_; }
      else if (using_named_ && expecting_val_) { order_[Nreq_ + working_opt_] = nparsed_; }
      nparsed_++;
      expecting_val_ = false;
      return valid_;
    }
    /*! @biref attempt to parse the next argument as a name (key) */
    bool parseNextAsKey(const String& key) {
      if (!valid_) {}
      else if ((nparsed_ < Nreq_) || (nparsed_-nkeys_ == Nreq_+Nopt_)) { valid_ = false; }
      else if (using_named_ && expecting_val_) { valid_ = false; }
      else if ((working_opt_ = find(key)) == keys_.size()) { valid_ = false; }
      else { using_named_ = true; expecting_val_ = true; nkeys_++; nparsed_++; }
      return valid_;
    }
    String toString(const String& method_name="f") const {
      int req_begin = 0, req_end = 0, opt_begin = 0, opt_end = 0;
      std::ostringstream s;
      // f(...)
      s << method_name << "(";
      // required arguments
      req_begin = s.str().size();
      for (size_t n = 0; n < Nreq_; ++n) { s << "src" << n+1 << (n != Nreq_-1 ? ", " : ""); }
      req_end = s.str().size();
      if (Nreq_ && Nopt_) s << ", ";
      // optional arguments
      opt_begin = s.str().size();
      for (size_t n = 0; n < keys_.size(); ++n) { s << "'" << keys_[n] << "', " << keys_[n] << (n != Nopt_-1 ? ", " : ""); }
      opt_end = s.str().size();
      s << ");";
      if (Nreq_ + Nopt_ == 0) return s.str();
      // underscores
      String under = String(req_begin, ' ') + String(req_end-req_begin, '-')
                   + String(std::max(opt_begin-req_end,0), ' ') + String(opt_end-opt_begin, '-');
      s << "\n" << under;
      // required and optional sets
      String req_set(req_end-req_begin, ' ');
      String opt_set(opt_end-opt_begin, ' ');
      if (!req_set.empty() && req_set.size() < 8) req_set.replace((req_set.size()-3)/2, 3, "req");
      if (req_set.size() > 7) req_set.replace((req_set.size()-8)/2, 8, "required");
      if (!opt_set.empty() && opt_set.size() < 8) opt_set.replace((opt_set.size()-3)/2, 3, "opt");
      if (opt_set.size() > 7) opt_set.replace((opt_set.size()-8)/2, 8, "optional");
      String set = String(req_begin, ' ') + req_set + String(std::max(opt_begin-req_end,0), ' ') + opt_set;
      s << "\n" << set;
      return s.str();
    }
  };
  /*! @brief given an input and output vector of arguments, and a variant spec, sort */
  void sortArguments(Variant& v, MxArrayVector& in, MxArrayVector& out) {
    // allocate the output array with ALL arguments
    out.resize(v.size());
    // reorder the inputs based on the variant ordering
    for (size_t n = 0; n < v.size(); ++n) {
      if (v.order(n) >= in.size()) continue;
      swap(in[v.order(n)], out[n]);
    }
  }
  VariantVector variants_;
  String valid_;
  String method_name_;
public:
  ArgumentParser(const String& method_name) : method_name_(method_name) {}

  /*! @brief add a function call variant to the parser
   *
   * Adds a function-call signature to the parser. The function call *must* be
   * unique either in its number of arguments, or in the named-syntax.
   * Currently this function does not check whether that invariant stands true.
   *
   * This function is variadic. If should be called as follows:
   *  addVariant(2, 2, 'opt_1_name', 'opt_2_name');
   */
  void addVariant(const String& name, size_t nreq, size_t nopt = 0, ...) {
    StringVector keys;
    va_list opt;
    va_start(opt, nopt);
    for (size_t n = 0; n < nopt; ++n) keys.push_back(va_arg(opt, const char*));
    addVariant(name, nreq, nopt, keys);
  }
  void addVariant(const String& name, size_t nreq, size_t nopt, StringVector keys) {
    variants_.push_back(Variant(name, nreq, nopt, keys));
  }

  /*! @brief check if the valid variant is the key name */
  bool variantIs(const String& name) {
    return name.compare(valid_) == 0;
  }

  /*! @brief parse a vector of input arguments
   *
   * This method parses a vector of input arguments, attempting to match them
   * to a Variant spec. For each input, the method attempts to cull any
   * Variants which don't match the given inputs so far.
   *
   * Once all inputs have been parsed, if there is one unique spec remaining,
   * the output MxArray vector gets populated with the arguments, with named
   * arguments removed. Any optional arguments that have not been encountered
   * are set to an empty array.
   *
   * If multiple variants or no variants match the given call, an error
   * message is emitted
   */
  MxArrayVector parse(const MxArrayVector& inputs) {
    // allocate the outputs
    String variant_string;
    MxArrayVector outputs;
    VariantVector candidates = variants_;

    // iterate over the inputs, attempting to match a variant
    for (MxArrayVector::const_iterator input = inputs.begin(); input != inputs.end(); ++input) {
      String name = input->isString() ? input->toString() : String();
      for (VariantVector::iterator candidate = candidates.begin(); candidate < candidates.end(); ++candidate) {
        candidate->exist(name) ? candidate->parseNextAsKey(name) : candidate->parseNextAsValue();
      }
    }

    // make sure the candidates have been fulfilled
    for (VariantVector::iterator candidate = candidates.begin(); candidate < candidates.end(); ++candidate) {
      if (!candidate->fulfilled()) candidate = candidates.erase(candidate)--;
    }

    // if there is not a unique candidate, throw an error
    for (VariantVector::iterator variant = variants_.begin(); variant != variants_.end(); ++variant) {
      variant_string += "\n" + variant->toString(method_name_);
    }

    // if there is not a unique candidate, throw an error
    if (candidates.size()  > 1) {
      error(String("Call to method is ambiguous. Valid variants are:")
        .append(variant_string).append("\nUse named arguments to disambiguate call"));
    }
    if (candidates.size() == 0) {
      error(String("No matching method signatures for given arguments. Valid variants are:").append(variant_string));
    }

    // Unique candidate!
    valid_ = candidates[0].name();
    sortArguments(candidates[0], const_cast<MxArrayVector&>(inputs), outputs);
    return outputs;
  }
};

} // namespace matlab

#endif
