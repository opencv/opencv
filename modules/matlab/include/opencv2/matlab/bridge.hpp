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
#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mxarray.hpp"
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/photo.hpp>

namespace cv {
namespace bridge {

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
typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<cv::DMatch> > vector_vector_DMatch;
typedef std::vector<cv::Rect> vector_Rect;
typedef std::vector<cv::KeyPoint> vector_KeyPoint;
typedef cv::Ptr<cv::StereoBM> Ptr_StereoBM;
typedef cv::Ptr<cv::StereoSGBM> Ptr_StereoSGBM;
typedef cv::Ptr<cv::FeatureDetector> Ptr_FeatureDetector;
typedef cv::Ptr<CLAHE> Ptr_CLAHE;
typedef cv::Ptr<LineSegmentDetector> Ptr_LineSegmentDetector;
typedef cv::Ptr<AlignMTB> Ptr_AlignMTB;
typedef cv::Ptr<CalibrateDebevec> Ptr_CalibrateDebevec;
typedef cv::Ptr<CalibrateRobertson> Ptr_CalibrateRobertson;
typedef cv::Ptr<MergeDebevec> Ptr_MergeDebevec;
typedef cv::Ptr<MergeMertens> Ptr_MergeMertens;
typedef cv::Ptr<MergeRobertson> Ptr_MergeRobertson;
typedef cv::Ptr<Tonemap> Ptr_Tonemap;
typedef cv::Ptr<TonemapDrago> Ptr_TonemapDrago;
typedef cv::Ptr<TonemapDurand> Ptr_TonemapDurand;
typedef cv::Ptr<TonemapMantiuk> Ptr_TonemapMantiuk;
typedef cv::Ptr<TonemapReinhard> Ptr_TonemapReinhard;


// ----------------------------------------------------------------------------
//                          PREDECLARATIONS
// ----------------------------------------------------------------------------
class Bridge;
typedef std::vector<Bridge> BridgeVector;

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const cv::Mat& src, matlab::MxArray& dst);

template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const matlab::MxArray& src, cv::Mat& dst);




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
 * relative to Matlab's type. That is, Bridge always stores an matlab::MxArray object
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
  matlab::MxArray ptr_;
public:
  // bridges are default constructible
  Bridge() {}
  virtual ~Bridge() {}

  // --------------------------------------------------------------------------
  //                         Bridge Properties
  // --------------------------------------------------------------------------
  bool empty() const { return ptr_.empty(); }

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
      matlab::error(std::string("Expected class ").append(std::string(name))
                        .append(" but was given ").append(ptr_.className()));
    }
    // get the instance field
    matlab::MxArray inst = ptr_.field("inst_");
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
      matlab::error("Incorrect pointer type stored for architecture");
    }

    // finally check if the object is NULL
    matlab::conditionalError(obj, std::string("Object ").append(std::string(name)).append(std::string(" is NULL")));
    return obj;
  }


  // --------------------------------------------------------------------------
  //                           MATLAB TYPES
  // --------------------------------------------------------------------------
  Bridge& operator=(const mxArray* obj) { ptr_ = obj; return *this; }
  Bridge& operator=(const matlab::MxArray& obj) { ptr_ = obj; return *this; }
  Bridge(const matlab::MxArray& obj) : ptr_(obj) {}
  Bridge(const mxArray* obj) : ptr_(obj) {}
  matlab::MxArray toMxArray() { return ptr_; }


  // --------------------------------------------------------------------------
  //                         MATRIX CONVERSIONS
  // --------------------------------------------------------------------------
  Bridge& operator=(const cv::Mat& mat);
  cv::Mat toMat() const;
  operator cv::Mat() const { return toMat(); }

  template <typename Scalar>
  static matlab::MxArray FromMat(const cv::Mat& mat) {
    matlab::MxArray arr(mat.rows, mat.cols, mat.channels(), matlab::Traits<Scalar>::ScalarType);
    switch (mat.depth()) {
      case CV_8U:  deepCopyAndTranspose<uint8_t,  Scalar>(mat, arr); break;
      case CV_8S:  deepCopyAndTranspose<int8_t,   Scalar>(mat, arr); break;
      case CV_16U: deepCopyAndTranspose<uint16_t, Scalar>(mat, arr); break;
      case CV_16S: deepCopyAndTranspose<int16_t,  Scalar>(mat, arr); break;
      case CV_32S: deepCopyAndTranspose<int32_t,  Scalar>(mat, arr); break;
      case CV_32F: deepCopyAndTranspose<float,    Scalar>(mat, arr); break;
      case CV_64F: deepCopyAndTranspose<double,   Scalar>(mat, arr); break;
      default: matlab::error("Attempted to convert from unknown class");
    }
    return arr;
  }

  template <typename Scalar>
  cv::Mat toMat() const {
    cv::Mat mat(ptr_.rows(), ptr_.cols(), CV_MAKETYPE(cv::DataType<Scalar>::type, ptr_.channels()));
    switch (ptr_.ID()) {
      case mxINT8_CLASS:    deepCopyAndTranspose<int8_t,   Scalar>(ptr_, mat); break;
      case mxUINT8_CLASS:   deepCopyAndTranspose<uint8_t,  Scalar>(ptr_, mat); break;
      case mxINT16_CLASS:   deepCopyAndTranspose<int16_t,  Scalar>(ptr_, mat); break;
      case mxUINT16_CLASS:  deepCopyAndTranspose<uint16_t, Scalar>(ptr_, mat); break;
      case mxINT32_CLASS:   deepCopyAndTranspose<int32_t,  Scalar>(ptr_, mat); break;
      case mxUINT32_CLASS:  deepCopyAndTranspose<uint32_t, Scalar>(ptr_, mat); break;
      case mxINT64_CLASS:   deepCopyAndTranspose<int64_t,  Scalar>(ptr_, mat); break;
      case mxUINT64_CLASS:  deepCopyAndTranspose<uint64_t, Scalar>(ptr_, mat); break;
      case mxSINGLE_CLASS:  deepCopyAndTranspose<float,    Scalar>(ptr_, mat); break;
      case mxDOUBLE_CLASS:  deepCopyAndTranspose<double,   Scalar>(ptr_, mat); break;
      case mxCHAR_CLASS:    deepCopyAndTranspose<char,     Scalar>(ptr_, mat); break;
      case mxLOGICAL_CLASS: deepCopyAndTranspose<int8_t,   Scalar>(ptr_, mat); break;
      default: matlab::error("Attempted to convert from unknown class");
    }
    return mat;
  }



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

  // ------------------------ vector_vector_char ------------------------------
  Bridge& operator=(const vector_vector_char& ) { return *this; }
  vector_vector_char toVectorVectorChar() { return vector_vector_char(); }
  operator vector_vector_char() { return toVectorVectorChar(); }

  // ------------------------ vector_vector_DMatch ---------------------------
  Bridge& operator=(const vector_vector_DMatch& ) { return *this; }
  vector_vector_DMatch toVectorVectorDMatch() { return vector_vector_DMatch(); }
  operator vector_vector_DMatch() { return toVectorVectorDMatch(); }




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

  // ---------------------------   Ptr_CLAHE   --------------------------------
  Bridge& operator=(const Ptr_CLAHE& ) { return *this; }
  Ptr_CLAHE toPtrCLAHE() { return Ptr_CLAHE(); }
  operator Ptr_CLAHE() { return toPtrCLAHE(); }

  // ---------------------------   Ptr_LineSegmentDetector   ------------------
  Bridge& operator=(const Ptr_LineSegmentDetector& ) { return *this; }
  Ptr_LineSegmentDetector toPtrLineSegmentDetector() { return Ptr_LineSegmentDetector(); }
  operator Ptr_LineSegmentDetector() { return toPtrLineSegmentDetector(); }

  // ---------------------------   Ptr_AlignMTB   -----------------------------
  Bridge& operator=(const Ptr_AlignMTB& ) { return *this; }
  Ptr_AlignMTB toPtrAlignMTB() { return Ptr_AlignMTB(); }
  operator Ptr_AlignMTB() { return toPtrAlignMTB(); }

  // ---------------------------   Ptr_CalibrateDebevec   -------------------
  Bridge& operator=(const Ptr_CalibrateDebevec& ) { return *this; }
  Ptr_CalibrateDebevec toPtrCalibrateDebevec() { return Ptr_CalibrateDebevec(); }
  operator Ptr_CalibrateDebevec() { return toPtrCalibrateDebevec(); }

  // ---------------------------   Ptr_CalibrateRobertson   -------------------
  Bridge& operator=(const Ptr_CalibrateRobertson& ) { return *this; }
  Ptr_CalibrateRobertson toPtrCalibrateRobertson() { return Ptr_CalibrateRobertson(); }
  operator Ptr_CalibrateRobertson() { return toPtrCalibrateRobertson(); }

  // ---------------------------   Ptr_MergeDebevec   -----------------------
  Bridge& operator=(const Ptr_MergeDebevec& ) { return *this; }
  Ptr_MergeDebevec toPtrMergeDebevec() { return Ptr_MergeDebevec(); }
  operator Ptr_MergeDebevec() { return toPtrMergeDebevec(); }

  // ---------------------------   Ptr_MergeMertens   -----------------------
  Bridge& operator=(const Ptr_MergeMertens& ) { return *this; }
  Ptr_MergeMertens toPtrMergeMertens() { return Ptr_MergeMertens(); }
  operator Ptr_MergeMertens() { return toPtrMergeMertens(); }

  // ---------------------------   Ptr_MergeRobertson   -----------------------
  Bridge& operator=(const Ptr_MergeRobertson& ) { return *this; }
  Ptr_MergeRobertson toPtrMergeRobertson() { return Ptr_MergeRobertson(); }
  operator Ptr_MergeRobertson() { return toPtrMergeRobertson(); }

  // ---------------------------   Ptr_Tonemap   ------------------------------
  Bridge& operator=(const Ptr_Tonemap& ) { return *this; }
  Ptr_Tonemap toPtrTonemap() { return Ptr_Tonemap(); }
  operator Ptr_Tonemap() { return toPtrTonemap(); }

  // ---------------------------   Ptr_TonemapDrago   -------------------------
  Bridge& operator=(const Ptr_TonemapDrago& ) { return *this; }
  Ptr_TonemapDrago toPtrTonemapDrago() { return Ptr_TonemapDrago(); }
  operator Ptr_TonemapDrago() { return toPtrTonemapDrago(); }

  // ---------------------------   Ptr_TonemapDurand   ------------------------
  Bridge& operator=(const Ptr_TonemapDurand& ) { return *this; }
  Ptr_TonemapDurand toPtrTonemapDurand() { return Ptr_TonemapDurand(); }
  operator Ptr_TonemapDurand() { return toPtrTonemapDurand(); }

  // ---------------------------   Ptr_TonemapMantiuk   -----------------------
  Bridge& operator=(const Ptr_TonemapMantiuk& ) { return *this; }
  Ptr_TonemapMantiuk toPtrTonemapMantiuk() { return Ptr_TonemapMantiuk(); }
  operator Ptr_TonemapMantiuk() { return toPtrTonemapMantiuk(); }

  // ---------------------------   Ptr_TonemapReinhard   ----------------------
  Bridge& operator=(const Ptr_TonemapReinhard& ) { return *this; }
  Ptr_TonemapReinhard toPtrTonemapReinhard() { return Ptr_TonemapReinhard(); }
  operator Ptr_TonemapReinhard() { return toPtrTonemapReinhard(); }
}; // class Bridge



// --------------------------------------------------------------------------
//                           SPECIALIZATIONS
// --------------------------------------------------------------------------

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
matlab::MxArray Bridge::FromMat<matlab::InheritType>(const cv::Mat& mat) {
  switch (mat.depth()) {
    case CV_8U:  return FromMat<uint8_t>(mat);
    case CV_8S:  return FromMat<int8_t>(mat);
    case CV_16U: return FromMat<uint16_t>(mat);
    case CV_16S: return FromMat<int16_t>(mat);
    case CV_32S: return FromMat<int32_t>(mat);
    case CV_32F: return FromMat<double>(mat); //NOTE: Matlab uses double as native type!
    case CV_64F: return FromMat<double>(mat);
    default: matlab::error("Attempted to convert from unknown class");
  }
  return matlab::MxArray();
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
cv::Mat Bridge::toMat<matlab::InheritType>() const {
  switch (ptr_.ID()) {
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
    default: matlab::error("Attempted to convert from unknown class");
  }
  return cv::Mat();
}

Bridge& Bridge::operator=(const cv::Mat& mat) { ptr_ = FromMat<matlab::InheritType>(mat); return *this; }
cv::Mat Bridge::toMat() const { return toMat<matlab::InheritType>(); }


// ----------------------------------------------------------------------------
//                            MATRIX TRANSPOSE
// ----------------------------------------------------------------------------


template <typename InputScalar, typename OutputScalar>
void deepCopyAndTranspose(const cv::Mat& in, matlab::MxArray& out) {
  matlab::conditionalError(static_cast<size_t>(in.rows) == out.rows(), "Matrices must have the same number of rows");
  matlab::conditionalError(static_cast<size_t>(in.cols) == out.cols(), "Matrices must have the same number of cols");
  matlab::conditionalError(static_cast<size_t>(in.channels()) == out.channels(), "Matrices must have the same number of channels");
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
void deepCopyAndTranspose(const matlab::MxArray& in, cv::Mat& out) {
  matlab::conditionalError(in.rows() == static_cast<size_t>(out.rows), "Matrices must have the same number of rows");
  matlab::conditionalError(in.cols() == static_cast<size_t>(out.cols), "Matrices must have the same number of cols");
  matlab::conditionalError(in.channels() == static_cast<size_t>(out.channels()), "Matrices must have the same number of channels");
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



} // namespace bridge
} // namespace cv

#endif
