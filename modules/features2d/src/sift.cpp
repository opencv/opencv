/* 
 * Here:
 * 1.) SIFT imlementation of Andrea Vedaldi
 * 2.) wrapper of Vedaldi`s SIFT
 */

/****************************************************************************************\
  1.) Implementation of SIFT taken from http://www.vlfeat.org/~vedaldi/code/siftpp.html 
\****************************************************************************************/

// AUTORIGHTS
// Copyright (c) 2006 The Regents of the University of California
// All Rights Reserved.
//
// Created by Andrea Vedaldi (UCLA VisionLab)
//
// Permission to use, copy, modify, and distribute this software and its
// documentation for educational, research and non-profit purposes,
// without fee, and without a written agreement is hereby granted,
// provided that the above copyright notice, this paragraph and the
// following three paragraphs appear in all copies.
//
// This software program and documentation are copyrighted by The Regents
// of the University of California. The software program and
// documentation are supplied "as is", without any accompanying services
// from The Regents. The Regents does not warrant that the operation of
// the program will be uninterrupted or error-free. The end-user
// understands that the program was developed for research purposes and
// is advised not to rely exclusively on the program for any reason.
//
// This software embodies a method for which the following patent has
// been issued: "Method and apparatus for identifying scale invariant
// features in an image and use of same for locating an object in an
// image," David G. Lowe, US Patent 6,711,293 (March 23,
// 2004). Provisional application filed March 8, 1999. Asignee: The
// University of British Columbia.
//
// IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
// FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
// INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
// CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
// BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
// MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

#include "precomp.hpp"

#ifdef __arm__
#define ARM_NO_SIFT
#endif

#ifdef ANDROID
#undef  ARM_NO_SIFT
#endif //ANDROID

#ifdef ARM_NO_SIFT

static inline void throw_nosift() { CV_Error(CV_StsBadFunc, "The library is compiled under ARM without SIFT support"); }

cv::SIFT::CommonParams::CommonParams() { }
cv::SIFT::CommonParams::CommonParams( int, int, int, int ) { throw_nosift(); }
cv::SIFT::DetectorParams::DetectorParams() { throw_nosift(); }
cv::SIFT::DetectorParams::DetectorParams( double, double ) { throw_nosift(); }
cv::SIFT::DescriptorParams::DescriptorParams() { throw_nosift(); }
cv::SIFT::DescriptorParams::DescriptorParams( double, bool, bool ) { throw_nosift(); }
cv::SIFT::SIFT() { throw_nosift(); }
cv::SIFT::SIFT( double, double, int, int, int, int ) { throw_nosift(); }
cv::SIFT::SIFT( double, bool, bool, int, int, int, int ) { throw_nosift(); }
cv::SIFT::SIFT( const CommonParams&, const DetectorParams&, const DescriptorParams& ) { throw_nosift(); }
int cv::SIFT::descriptorSize() const {  throw_nosift(); return 0; }
void cv::SIFT::operator()( const Mat&, const Mat&, vector<KeyPoint>& ) const { throw_nosift(); }
void cv::SIFT::operator()( const Mat&, const Mat&, vector<KeyPoint>&, Mat&, bool ) const { throw_nosift(); }
cv::SIFT::CommonParams cv::SIFT::getCommonParams() const {  throw_nosift(); return cv::SIFT::CommonParams(); }
cv::SIFT::DetectorParams cv::SIFT::getDetectorParams() const {  throw_nosift(); return cv::SIFT::DetectorParams(); }
cv::SIFT::DescriptorParams cv::SIFT::getDescriptorParams() const {  throw_nosift(); return cv::SIFT::DescriptorParams(); }

#else // with SIFT

#include <iostream>
#include <limits>

#define log2(a) (log((a))/CV_LOG2)

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable: 4100 4244 4267 4305)
#endif

/*
 * from sift.hpp of original code
 */
#if defined (VL_USEFASTMATH)
#if defined (VL_MAC)
#define VL_FASTFLOAT float
#else
#define VL_FASTFLOAT double
#endif
#else
#define VL_FASTFLOAT float
#endif

/** @brief VisionLab namespace */
namespace VL {

/** @brief Pixel data type */
typedef float pixel_t ;

/** @brief Floating point data type
 **
 ** Although floats are precise enough for this applicatgion, on Intel
 ** based architecture using doubles for floating point computations
 ** turns out to be much faster.
 **/
typedef VL_FASTFLOAT float_t ;

/** @brief 32-bit floating data type */
typedef float float32_t ;

/** @brief 64-bit floating data type */
typedef double float64_t ;

/** @brief 32-bit integer data type */
typedef int int32_t ;

/** @brief 64-bit integer data type */
typedef long long int int64_t ;

/** @brief 32-bit unsigned integer data type */
typedef int uint32_t ;

/** @brief 8-bit unsigned integer data type */
typedef char unsigned uint8_t ;

/** @name Fast math
 **
 ** We provide approximate mathematical functions. These are usually
 ** rather faster than the corresponding standard library functions.
 **/
/*@{*/
float   fast_resqrt(float x) ;
double  fast_resqrt(double x) ;
float_t fast_expn(float_t x) ;
float_t fast_abs(float_t x) ;
float_t fast_mod_2pi(float_t x) ;
float_t fast_atan2(float_t y, float_t x) ;
float_t fast_sqrt(float_t x) ;
int32_t fast_floor(float_t x) ;
/*@}*/

/** @brief PGM buffer descriptor
 **
 ** The structure describes a gray scale image and it is used by the
 ** PGM input/output functions. The fileds are self-explanatory.
 **/
/*@{*/
struct PgmBuffer
{
  int width ;     ///< Image width
  int height ;    ///< Image hegith
  pixel_t* data ; ///< Image data
} ;
/*@}*/

/** @brief SIFT filter
 **
 ** This class is a filter computing the Scale Invariant Feature
 ** Transform (SIFT).
 **/
class Sift
{

public:

  /** @brief SIFT keypoint
   **
   ** A SIFT keypoint is charactedized by a location x,y and a scale
   ** @c sigma. The scale is obtained from the level index @c s and
   ** the octave index @c o through a simple formula (see the PDF
   ** documentation).
   **
   ** In addition to the location, scale indexes and scale, we also
   ** store the integer location and level. The integer location is
   ** unnormalized, i.e. relative to the resolution of the octave
   ** containing the keypoint (octaves are downsampled).
   **/
  struct Keypoint
  {
    int o ;    ///< Keypoint octave index

    int ix ;   ///< Keypoint integer X coordinate (unnormalized)
    int iy ;   ///< Keypoint integer Y coordinate (unnormalized)
    int is ;   ///< Keypoint integer scale indiex

    float_t x  ;  ///< Keypoint fractional X coordinate
    float_t y  ;  ///< Keypoint fractional Y coordinate
    float_t s ;   ///< Keypoint fractional scale index

    float_t sigma ;  ///< Keypoint scale
  } ;

  typedef std::vector<Keypoint>     Keypoints ;          ///< Keypoint list datatype
  typedef Keypoints::iterator       KeypointsIter ;      ///< Keypoint list iter datatype
  typedef Keypoints::const_iterator KeypointsConstIter ; ///< Keypoint list const iter datatype

#undef _S
  /** @brief Constructors and destructors */
  /*@{*/
  Sift(const pixel_t* _im_pt, int _width, int _height,
       float_t _sigman,
       float_t _sigma0,
       int _O, int _S,
       int _omin, int _smin, int _smax) ;
  ~Sift() ;
  /*@}*/

  void process(const pixel_t* _im_pt, int _width, int _height) ;

  /** @brief Querying the Gaussian scale space */
  /*@{*/
  VL::pixel_t* getOctave(int o) ;
  VL::pixel_t* getLevel(int o, int s) ;
  int          getWidth() const ;
  int          getHeight() const ;
  int          getOctaveWidth(int o) const ;
  int          getOctaveHeight(int o) const ;
  VL::float_t  getOctaveSamplingPeriod(int o) const ;
  VL::float_t  getScaleFromIndex(VL::float_t o, VL::float_t s) const ;
  Keypoint     getKeypoint(VL::float_t x, VL::float_t y, VL::float_t s) const ;
  /*@}*/

  /** @brief Descriptor parameters */
  /*@{*/
  bool getNormalizeDescriptor() const ;
  void setNormalizeDescriptor(bool) ;
  void setMagnification(VL::float_t) ;
  VL::float_t getMagnification() const ;
  /*@}*/

  /** @brief Detector and descriptor */
  /*@{*/
  void detectKeypoints(VL::float_t threshold, VL::float_t edgeThreshold) ;
  int computeKeypointOrientations(VL::float_t angles [4], Keypoint keypoint) ;
  void computeKeypointDescriptor(VL::float_t* descr_pt, Keypoint keypoint, VL::float_t angle) ;
  KeypointsIter keypointsBegin() ;
  KeypointsIter keypointsEnd() ;
  /*@}*/

private:
  void prepareBuffers() ;
  void freeBuffers() ;
  void smooth(VL::pixel_t       * dst,
              VL::pixel_t       * temp,
              VL::pixel_t const * src, int width, int height,
              VL::float_t s) ;

  void prepareGrad(int o) ;

  // scale space parameters
  VL::float_t sigman ;
  VL::float_t sigma0 ;
  VL::float_t sigmak ;

  int O ;
  int S ;
  int omin ;
  int smin ;
  int smax ;

  int width ;
  int height ;

  // descriptor parameters
  VL::float_t magnif ;
  bool        normalizeDescriptor ;

  // buffers
  VL::pixel_t*  temp ;
  int           tempReserved ;
  bool          tempIsGrad  ;
  int           tempOctave ;
  VL::pixel_t** octaves ;

  VL::pixel_t*  filter ;
  int           filterReserved ;

  Keypoints keypoints ;
} ;

}

/*
 * from sift.ipp of original code
 */
namespace VL
{

namespace Detail
{
extern int const expnTableSize ;
extern VL::float_t const expnTableMax ;
extern VL::float_t expnTable [] ;
}

/** @brief Get width of source image
 ** @result width.
 **/
inline
int
Sift::getWidth() const
{
  return width ;
}

/** @brief Get height of source image
 ** @result height.
 **/
inline
int
Sift::getHeight() const
{
  return height ;
}

/** @brief Get width of an octave
 ** @param o octave index.
 ** @result width of octave @a o.
 **/
inline
int
Sift::getOctaveWidth(int o) const
{
  assert( omin <= o && o < omin + O ) ;
  return (o >= 0) ? (width >> o) : (width << -o) ;
}

/** @brief Get height of an octave
 ** @param o octave index.
 ** @result height of octave @a o.
 **/
inline
int
Sift::getOctaveHeight(int o) const
{
  assert( omin <= o && o < omin + O ) ;
  return (o >= 0) ? (height >> o) : (height << -o) ;
}

/** @brief Get octave
 ** @param o octave index.
 ** @return pointer to octave @a o.
 **/
inline
VL::pixel_t *
Sift::getOctave(int o)
{
  assert( omin <= o && o < omin + O ) ;
  return octaves[o-omin] ;
}

/** @brief Get level
 ** @param o octave index.
 ** @param s level index.
 ** @result pointer to level @c (o,s).
 **/
inline
VL::pixel_t *
Sift::getLevel(int o, int s)
{
  assert( omin <= o && o <  omin + O ) ;
  assert( smin <= s && s <= smax     ) ;
  return octaves[o - omin] +
    getOctaveWidth(o)*getOctaveHeight(o) * (s-smin) ;
}

/** @brief Get octave sampling period
 ** @param o octave index.
 ** @result Octave sampling period (in pixels).
 **/
inline
VL::float_t
Sift::getOctaveSamplingPeriod(int o) const
{
  return (o >= 0) ? (1 << o) : 1.0f / (1 << -o) ;
}

/** @brief Convert index into scale
 ** @param o octave index.
 ** @param s scale index.
 ** @return scale.
 **/
inline
VL::float_t
Sift::getScaleFromIndex(VL::float_t o, VL::float_t s) const
{
  return sigma0 * powf( 2.0f, o + s / S ) ;
}

/** @brief Get keypoint list begin
 ** @return iterator to the beginning.
 **/
inline
Sift::KeypointsIter
Sift::keypointsBegin()
{
  return keypoints.begin() ;
}

/** @brief Get keypoint list end
 ** @return iterator to the end.
 **/
inline
Sift::KeypointsIter
Sift::keypointsEnd()
{
  return keypoints.end() ;
}

/** @brief Set normalize descriptor flag */
inline
void
Sift::setNormalizeDescriptor(bool flag)
{
  normalizeDescriptor = flag ;
}

/** @brief Get normalize descriptor flag */
inline
bool
Sift::getNormalizeDescriptor() const
{
  return normalizeDescriptor ;
}

/** @brief Set descriptor magnification */
inline
void
Sift::setMagnification(VL::float_t _magnif)
{
  magnif = _magnif ;
}

/** @brief Get descriptor magnification */
inline
VL::float_t
Sift::getMagnification() const
{
  return magnif ;
}

/** @brief Fast @ exp(-x)
 **
 ** The argument must be in the range 0-25.0 (bigger arguments may be
 ** truncated to zero).
 **
 ** @param x argument.
 ** @return @c exp(-x)
 **/
inline
VL::float_t
fast_expn(VL::float_t x)
{
  assert(VL::float_t(0) <= x && x <= Detail::expnTableMax) ;
#ifdef VL_USEFASTMATH
  x *= Detail::expnTableSize / Detail::expnTableMax ;
  VL::int32_t i = fast_floor(x) ;
  VL::float_t r = x - i ;
  VL::float_t a = VL::Detail::expnTable[i] ;
  VL::float_t b = VL::Detail::expnTable[i+1] ;
  return a + r * (b - a) ;
#else
  return exp(-x) ;
#endif
}

/** @brief Fast @c mod(x,2pi)
 **
 ** The function quickly computes the value @c mod(x,2pi).
 **
 ** @remark The computation is fast only for arguments @a x which are
 ** small in modulus.
 **
 ** @remark For negative arguments, the semantic of the function is
 ** not equivalent to the standard library @c fmod function.
 **
 ** @param x function argument.
 ** @return @c mod(x,2pi)
 **/
inline
VL::float_t
fast_mod_2pi(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  while(x < VL::float_t(0)      ) x += VL::float_t(2*CV_PI) ;
  while(x > VL::float_t(2*CV_PI) ) x -= VL::float_t(2*CV_PI) ;
  return x ;
#else
  return (x>=0) ? std::fmod(x, VL::float_t(2*CV_PI))
    : 2*CV_PI + std::fmod(x, VL::float_t(2*CV_PI)) ;
#endif
}

/** @brief Fast @c (int) floor(x)
 ** @param x argument.
 ** @return @c float(x)
 **/
inline
int32_t
fast_floor(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x>=0)? int32_t(x) : std::floor(x) ;
  //  return int32_t( x - ((x>=0)?0:1) ) ;
#else
  return int32_t( std::floor(x) ) ;
#endif
}

/** @brief Fast @c abs(x)
 ** @param x argument.
 ** @return @c abs(x)
 **/
inline
VL::float_t
fast_abs(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x >= 0) ? x : -x ;
#else
  return std::fabs(x) ;
#endif
}

/** @brief Fast @c atan2
 ** @param x argument.
 ** @param y argument.
 ** @return Approximation of @c atan2(x).
 **/
inline
VL::float_t
fast_atan2(VL::float_t y, VL::float_t x)
{
#ifdef VL_USEFASTMATH

  /*
    The function f(r)=atan((1-r)/(1+r)) for r in [-1,1] is easier to
    approximate than atan(z) for z in [0,inf]. To approximate f(r) to
    the third degree we may solve the system

     f(+1) = c0 + c1 + c2 + c3 = atan(0) = 0
     f(-1) = c0 - c1 + c2 - c3 = atan(inf) = pi/2
     f(0)  = c0                = atan(1) = pi/4

    which constrains the polynomial to go through the end points and
    the middle point.

    We still miss a constrain, which might be simply a constarint on
    the derivative in 0. Instead we minimize the Linf error in the
    range [0,1] by searching for an optimal value of the free
    parameter. This turns out to correspond to the solution

     c0=pi/4, c1=-0.9675, c2=0, c3=0.1821

    which has maxerr = 0.0061 rad = 0.35 grad.
  */

  VL::float_t angle, r ;
  VL::float_t const c3 = 0.1821 ;
  VL::float_t const c1 = 0.9675 ;
  VL::float_t abs_y    = fast_abs(y) + VL::float_t(1e-10) ;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = VL::float_t(CV_PI/4.0) ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = VL::float_t(3*CV_PI/4.0) ;
  }
  angle += (c3*r*r - c1) * r ;
  return (y < 0) ? -angle : angle ;
#else
  return std::atan2(y,x) ;
#endif
}

/** @brief Fast @c resqrt
 ** @param x argument.
 ** @return Approximation to @c resqrt(x).
 **/
inline
float
fast_resqrt(float x)
{
#ifdef VL_USEFASTMATH
  // Works if VL::float_t is 32 bit ...
  union {
    float x ;
    VL::int32_t i ;
  } u ;
  float xhalf = float(0.5) * x ;
  u.x = x ;                               // get bits for floating value
  u.i = 0x5f3759df - (u.i>>1);            // gives initial guess y0
  //u.i = 0xdf59375f - (u.i>>1);          // gives initial guess y0
  u.x = u.x*(float(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  u.x = u.x*(float(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  return u.x ;
#else
  return float(1.0) / std::sqrt(x) ;
#endif
}

/** @brief Fast @c resqrt
 ** @param x argument.
 ** @return Approximation to @c resqrt(x).
 **/
inline
double
fast_resqrt(double x)
{
#ifdef VL_USEFASTMATH
  // Works if double is 64 bit ...
  union {
    double x ;
    VL::int64_t i ;
  } u ;
  double xhalf = double(0.5) * x ;
  u.x = x ;                                // get bits for floating value
  u.i = 0x5fe6ec85e7de30daLL - (u.i>>1);   // gives initial guess y0
  u.x = u.x*(double(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  u.x = u.x*(double(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  return u.x ;
#else
  return double(1.0) / std::sqrt(x) ;
#endif
}

/** @brief Fast @c sqrt
 ** @param x argument.
 ** @return Approximation to @c sqrt(x).
 **/
inline
VL::float_t
fast_sqrt(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x < 1e-8) ? 0 : x * fast_resqrt(x) ;
#else
  return std::sqrt(x) ;
#endif
}

}

/*
 * from sift.tpp of original code
 */

template<typename T>
void
normalize(T* filter, int W)
{
  T  acc  = 0 ;
  T* iter = filter ;
  T* end  = filter + 2*W+1 ;
  while(iter != end) acc += *iter++ ;

  iter = filter ;
  while(iter != end) *iter++ /= acc ;
}

template<typename T>
void
convolve(T*       dst_pt,
         const T* src_pt, int M, int N,
         const T* filter_pt, int W)
{
  typedef T const TC ;
  // convolve along columns, save transpose
  // image is M by N
  // buffer is N by M
  // filter is (2*W+1) by 1
  for(int j = 0 ; j < N ; ++j) {

    int i = 0 ;

    // top
    for(; i <= std::min(W-1, M-1) ; ++i) {
      TC* start = src_pt ;
      TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
      TC* g     = filter_pt + W-i ;
      T   acc = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }

    // middle
    // run this for W <= i <= M-1-W, only if M >= 2*W+1
    for(; i <= M-1-W ; ++i) {
      TC* start = src_pt    + i-W ;
      TC* stop  = src_pt    + i+W + 1 ;
      TC* g     = filter_pt ;
      T   acc = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }

    // bottom
    // run this for M-W <= i <= M-1, only if M >= 2*W+1
    for(; i <= M-1 ; ++i) {
      TC* start = src_pt    + i-W ;
      TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
      TC* g     = filter_pt ;
      T   acc   = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }

    // next column
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}

// works with symmetric filters only
template<typename T>
void
nconvolve(T*       dst_pt,
          const T* src_pt, int M, int N,
          const T* filter_pt, int W,
          T*       scratch_pt )
{
  typedef T const TC ;

  for(int i = 0 ; i <= W ; ++i) {
    T   acc = 0.0 ;
    TC* iter = filter_pt + std::max(W-i,  0) ;
    TC* stop = filter_pt + std::min(M-1-i,W) + W + 1 ;
    while(iter != stop) acc += *iter++ ;
    scratch_pt [i] = acc ;
  }

 for(int j = 0 ; j < N ; ++j) {

   int i = 0 ;
   // top margin
   for(; i <= std::min(W, M-1) ; ++i) {
     TC* start = src_pt ;
     TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
     TC* g     = filter_pt + W-i ;
     T   acc = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc / scratch_pt [i] ;
     dst_pt += N ;
   }

   // middle
   for(; i <= M-1-W ; ++i) {
     TC* start = src_pt    + i-W ;
     TC* stop  = src_pt    + i+W + 1 ;
     TC* g     = filter_pt ;
     T   acc = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc ;
     dst_pt += N ;
   }

   // bottom
   for(; i <= M-1 ; ++i) {
     TC* start = src_pt    + i-W ;
     TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
     TC* g     = filter_pt ;
     T   acc   = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc / scratch_pt [M-1-i];
     dst_pt += N ;
   }

   // next column
   src_pt += M ;
   dst_pt -= M*N - 1 ;
 }
}

template<typename T>
void
econvolve(T*       dst_pt,
      const T* src_pt,    int M, int N,
      const T* filter_pt, int W)
{
  typedef T const TC ;
  // convolve along columns, save transpose
  // image is M by N
  // buffer is N by M
  // filter is (2*W+1) by 1
  for(int j = 0 ; j < N ; ++j) {
    for(int i = 0 ; i < M ; ++i) {
      T   acc = 0.0 ;
      TC* g = filter_pt ;
      TC* start = src_pt + (i-W) ;
      TC* stop  ;
      T   x ;

      // beginning
      stop = src_pt + std::max(0, i-W) ;
      x    = *stop ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; }

      // middle
      stop =  src_pt + std::min(M-1, i+W) ;
      while( start <  stop ) acc += (*g++) * (*start++) ;

      // end
      x  = *start ;
      stop = src_pt + (i+W) ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; }

      // save
      *dst_pt = acc ;
      dst_pt += N ;

      assert( g - filter_pt == 2*W+1 ) ;

    }
    // next column
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}

/*
 * from sift.cpp of original code
 */

extern "C" {
#if defined (VL_MAC)
#include<libgen.h>
#else
#include<string.h>
}
#endif

// on startup, pre-compute expn(x) = exp(-x)
namespace VL {
namespace Detail {

int const         expnTableSize = 256 ;
VL::float_t const expnTableMax  = VL::float_t(25.0) ;
VL::float_t       expnTable [ expnTableSize + 1 ] ;

struct buildExpnTable
{
  buildExpnTable() {
    for(int k = 0 ; k < expnTableSize + 1 ; ++k) {
      expnTable[k] = exp( - VL::float_t(k) / expnTableSize * expnTableMax ) ;
    }
  }
} _buildExpnTable ;

} }


namespace VL {

    // ===================================================================
//                                          Low level image operations
// -------------------------------------------------------------------

namespace Detail {

/** @brief Copy an image
 ** @param dst    output imgage buffer.
 ** @param src    input image buffer.
 ** @param width  input image width.
 ** @param height input image height.
 **/
void
copy(pixel_t* dst, pixel_t const* src, int width, int height)
{
  memcpy(dst, src, sizeof(pixel_t)*width*height)  ;
}

/** @brief Copy an image upsampling two times
 **
 ** The destination buffer must be at least as big as two times the
 ** input buffer. Bilinear interpolation is used.
 **
 ** @param dst     output imgage buffer.
 ** @param src     input image buffer.
 ** @param width   input image width.
 ** @param height  input image height.
 **/
void
copyAndUpsampleRows
(pixel_t* dst, pixel_t const* src, int width, int height)
{
  for(int y = 0 ; y < height ; ++y) {
    pixel_t b, a ;
    b = a = *src++ ;
    for(int x = 0 ; x < width-1 ; ++x) {
      b = *src++ ;
      *dst = a ;         dst += height ;
      *dst = 0.5*(a+b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }
}

/** @brief Copy and downasample an image
 **
 ** The image is downsampled @a d times, i.e. reduced to @c 1/2^d of
 ** its original size. The parameters @a width and @a height are the
 ** size of the input image. The destination image is assumed to be @c
 ** floor(width/2^d) pixels wide and @c floor(height/2^d) pixels high.
 **
 ** @param dst output imgage buffer.
 ** @param src input image buffer.
 ** @param width input image width.
 ** @param height input image height.
 ** @param d downsampling factor.
 **/
void
copyAndDownsample(pixel_t* dst, pixel_t const* src,
                  int width, int height, int d)
{
  for(int y = 0 ; y < height ; y+=d) {
    pixel_t const * srcrowp = src + y * width ;
    for(int x = 0 ; x < width - (d-1) ; x+=d) {
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}

}

/** @brief Smooth an image
 **
 ** The function convolves the image @a src by a Gaussian kernel of
 ** variance @a s and writes the result to @a dst. The function also
 ** needs a scratch buffer @a dst of the same size of @a src and @a
 ** dst.
 **
 ** @param dst output image buffer.
 ** @param temp scratch image buffer.
 ** @param src input image buffer.
 ** @param width width of the buffers.
 ** @param height height of the buffers.
 ** @param s standard deviation of the Gaussian kernel.
 **/
void
Sift::smooth
(pixel_t* dst, pixel_t* temp,
 pixel_t const* src, int width, int height,
 VL::float_t s)
{
  // make sure a buffer larege enough has been allocated
  // to hold the filter
  int W = int( ceil( VL::float_t(4.0) * s ) ) ;
  if( ! filter ) {
    filterReserved = 0 ;
  }

  if( filterReserved < W ) {
    filterReserved = W ;
    if( filter ) delete [] filter ;
    filter = new pixel_t [ 2* filterReserved + 1 ] ;
  }

  // pre-compute filter
  for(int j = 0 ; j < 2*W+1 ; ++j)
    filter[j] = VL::pixel_t
      (std::exp
       (VL::float_t
        (-0.5 * (j-W) * (j-W) / (s*s) ))) ;

  // normalize to one
  normalize(filter, W) ;

  // convolve
  econvolve(temp, src, width, height, filter, W) ;
  econvolve(dst, temp, height, width, filter, W) ;
}

// ===================================================================
//                                                     Sift(), ~Sift()
// -------------------------------------------------------------------

/** @brief Initialize Gaussian scale space parameters
 **
 ** @param _im_pt  Source image data
 ** @param _width  Soruce image width
 ** @param _height Soruce image height
 ** @param _sigman Nominal smoothing value of the input image.
 ** @param _sigma0 Base smoothing level.
 ** @param _O      Number of octaves.
 ** @param _S      Number of levels per octave.
 ** @param _omin   First octave.
 ** @param _smin   First level in each octave.
 ** @param _smax   Last level in each octave.
 **/
Sift::Sift(const pixel_t* _im_pt, int _width, int _height,
     VL::float_t _sigman,
     VL::float_t _sigma0,
     int _O, int _S,
     int _omin, int _smin, int _smax)
  : sigman( _sigman ),
    sigma0( _sigma0 ),
    O( _O ),
    S( _S ),
    omin( _omin ),
    smin( _smin ),
    smax( _smax ),

    magnif( 3.0f ),
    normalizeDescriptor( true ),

    temp( NULL ),
    octaves( NULL ),
    filter( NULL )
{
  process(_im_pt, _width, _height) ;
}

/** @brief Destroy SIFT filter.
 **/
Sift::~Sift()
{
  freeBuffers() ;
}

/** Allocate buffers. Buffer sizes depend on the image size and the
 ** value of omin.
 **/
void
Sift::
prepareBuffers()
{
  // compute buffer size
  int w = (omin >= 0) ? (width  >> omin) : (width  << -omin) ;
  int h = (omin >= 0) ? (height >> omin) : (height << -omin) ;
  int size = w*h* std::max
    ((smax - smin), 2*((smax+1) - (smin-2) +1)) ;

  if( temp && tempReserved == size ) return ;

  freeBuffers() ;

  // allocate
  temp           = new pixel_t [ size ] ;
  tempReserved   = size ;
  tempIsGrad     = false ;
  tempOctave     = 0 ;

  octaves = new pixel_t* [ O ] ;
  for(int o = 0 ; o < O ; ++o) {
    octaves[o] = new pixel_t [ (smax - smin + 1) * w * h ] ;
    w >>= 1 ;
    h >>= 1 ;
  }
}

/** @brief Free buffers.
 **
 ** This function releases any buffer allocated by prepareBuffers().
 **
 ** @sa prepareBuffers().
 **/
void
Sift::
freeBuffers()
{
  if( filter ) {
    delete [] filter ;
  }
  filter = 0 ;

  if( octaves ) {
    for(int o = 0 ; o < O ; ++o) {
      delete [] octaves[ o ] ;
    }
    delete [] octaves ;
  }
  octaves = 0 ;

  if( temp ) {
    delete [] temp ;
  }
  temp = 0  ;
}

// ===================================================================
//                                                         getKeypoint
// -------------------------------------------------------------------

/** @brief Get keypoint from position and scale
 **
 ** The function returns a keypoint with a given position and
 ** scale. Note that the keypoint structure contains fields that make
 ** sense only in conjunction with a specific scale space. Therefore
 ** the keypoint structure should be re-calculated whenever the filter
 ** is applied to a new image, even if the parameters @a x, @a y and
 ** @a sigma do not change.
 **
 ** @param x x coordinate of the center.
 ** @peram y y coordinate of the center.
 ** @param sigma scale.
 ** @return Corresponing keypoint.
 **/
Sift::Keypoint
Sift::getKeypoint(VL::float_t x, VL::float_t y, VL::float_t sigma) const
{

  /*
    The formula linking the keypoint scale sigma to the octave and
    scale index is

    (1) sigma(o,s) = sigma0 2^(o+s/S)

    for which

    (2) o + s/S = log2 sigma/sigma0 == phi.

    In addition to the scale index s (which can be fractional due to
    scale interpolation) a keypoint has an integer scale index is too
    (which is the index of the scale level where it was detected in
    the DoG scale space). We have the constraints:

    - o and is are integer

    - is is in the range [smin+1, smax-2  ]

    - o  is in the range [omin,   omin+O-1]

    - is = rand(s) most of the times (but not always, due to the way s
      is obtained by quadratic interpolation of the DoG scale space).

    Depending on the values of smin and smax, often (2) has multiple
    solutions is,o that satisfy all constraints.  In this case we
    choose the one with biggest index o (this saves a bit of
    computation).

    DETERMINING THE OCTAVE INDEX O

    From (2) we have o = phi - s/S and we want to pick the biggest
    possible index o in the feasible range. This corresponds to
    selecting the smallest possible index s. We write s = is + ds
    where in most cases |ds|<.5 (but in general |ds|<1). So we have

       o = phi - s/S,   s = is + ds ,   |ds| < .5 (or |ds| < 1).

    Since is is in the range [smin+1,smax-2], s is in the range
    [smin+.5,smax-1.5] (or [smin,smax-1]), the number o is an integer
    in the range phi+[-smax+1.5,-smin-.5] (or
    phi+[-smax+1,-smin]). Thus the maximum value of o is obtained for
    o = floor(phi-smin-.5) (or o = floor(phi-smin)).

    Finally o is clamped to make sure it is contained in the feasible
    range.

    DETERMINING THE SCALE INDEXES S AND IS

    Given o we can derive is by writing (2) as

      s = is + ds = S(phi - o).

    We then take is = round(s) and clamp its value to be in the
    feasible range.
  */

  int o,ix,iy,is ;
  VL::float_t s,phi ;

  phi = log2(sigma/sigma0) ;
  o   = fast_floor( phi -  (VL::float_t(smin)+.5)/S ) ;
  o   = std::min(o, omin+O-1) ;
  o   = std::max(o, omin    ) ;
  s   = S * (phi - o) ;

  is  = int(s + 0.5) ;
  is  = std::min(is, smax - 2) ;
  is  = std::max(is, smin + 1) ;

  VL::float_t per = getOctaveSamplingPeriod(o) ;
  ix = int(x / per + 0.5) ;
  iy = int(y / per + 0.5) ;

  Keypoint key ;
  key.o  = o ;

  key.ix = ix ;
  key.iy = iy ;
  key.is = is ;

  key.x = x ;
  key.y = y ;
  key.s = s ;

  key.sigma = sigma ;

  return key ;
}

// ===================================================================
//                                                           process()
// -------------------------------------------------------------------

/** @brief Compute Gaussian Scale Space
 **
 ** The method computes the Gaussian scale space of the specified
 ** image. The scale space data is managed internally and can be
 ** accessed by means of getOctave() and getLevel().
 **
 ** @remark Calling this method will delete the list of keypoints
 ** constructed by detectKeypoints().
 **
 ** @param _im_pt pointer to image data.
 ** @param _width image width.
 ** @param _height image height .
 **/
void
Sift::
process(const pixel_t* _im_pt, int _width, int _height)
{
  using namespace Detail ;

  width  = _width ;
  height = _height ;
  prepareBuffers() ;

  VL::float_t sigmak = powf(2.0f, 1.0 / S) ;
  VL::float_t dsigma0 = sigma0 * sqrt (1.0f - 1.0f / (sigmak*sigmak) ) ;

  // -----------------------------------------------------------------
  //                                                 Make pyramid base
  // -----------------------------------------------------------------
  if( omin < 0 ) {
    copyAndUpsampleRows(temp,       _im_pt, width,  height  ) ;
    copyAndUpsampleRows(octaves[0], temp,   height, 2*width ) ;

    for(int o = -1 ; o > omin ; --o) {
      copyAndUpsampleRows(temp,       octaves[0], width  << -o,    height << -o) ;
      copyAndUpsampleRows(octaves[0], temp,       height << -o, 2*(width  << -o)) ;             }

  } else if( omin > 0 ) {
    copyAndDownsample(octaves[0], _im_pt, width, height, 1 << omin) ;
  } else {
    copy(octaves[0], _im_pt, width, height) ;
  }

  {
    VL::float_t sa = sigma0 * powf(sigmak, smin) ;
    VL::float_t sb = sigman / powf(2.0f,   omin) ; // review this
    if( sa > sb ) {
      VL::float_t sd = sqrt ( sa*sa - sb*sb ) ;
      smooth( octaves[0], temp, octaves[0],
              getOctaveWidth(omin),
              getOctaveHeight(omin),
              sd ) ;
    }
  }

  // -----------------------------------------------------------------
  //                                                      Make octaves
  // -----------------------------------------------------------------
  for(int o = omin ; o < omin+O ; ++o) {
    // Prepare octave base
    if( o > omin ) {
      int sbest = std::min(smin + S, smax) ;
      copyAndDownsample(getLevel(o,   smin ),
                getLevel(o-1, sbest),
                getOctaveWidth(o-1),
                getOctaveHeight(o-1), 2 ) ;
      VL::float_t sa = sigma0 * powf(sigmak, smin      ) ;
      VL::float_t sb = sigma0 * powf(sigmak, sbest - S ) ;
      if(sa > sb ) {
        VL::float_t sd = sqrt ( sa*sa - sb*sb ) ;
        smooth( getLevel(o,0), temp, getLevel(o,0),
                getOctaveWidth(o), getOctaveHeight(o),
                sd ) ;
      }
    }

    // Make other levels
    for(int s = smin+1 ; s <= smax ; ++s) {
      VL::float_t sd = dsigma0 * powf(sigmak, s) ;
      smooth( getLevel(o,s), temp, getLevel(o,s-1),
              getOctaveWidth(o), getOctaveHeight(o),
              sd ) ;
    }
  }
}

/** @brief Sift detector
 **
 ** The function runs the SIFT detector on the stored Gaussian scale
 ** space (see process()). The detector consists in three steps
 **
 ** - local maxima detection;
 ** - subpixel interpolation;
 ** - rejection of weak keypoints (@a threhsold);
 ** - rejection of keypoints on edge-like structures (@a edgeThreshold).
 **
 ** As they are found, keypoints are added to an internal list.  This
 ** list can be accessed by means of the member functions
 ** getKeypointsBegin() and getKeypointsEnd(). The list is ordered by
 ** octave, which is usefult to speed-up computeKeypointOrientations()
 ** and computeKeypointDescriptor().
 **/
void
Sift::detectKeypoints(VL::float_t threshold, VL::float_t edgeThreshold)
{
  keypoints.clear() ;

  int nValidatedKeypoints = 0 ;

  // Process one octave per time
  for(int o = omin ; o < omin + O ; ++o) {

    int const xo = 1 ;
    int const yo = getOctaveWidth(o) ;
    int const so = getOctaveWidth(o) * getOctaveHeight(o) ;
    int const ow = getOctaveWidth(o) ;
    int const oh = getOctaveHeight(o) ;

    VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

    // -----------------------------------------------------------------
    //                                           Difference of Gaussians
    // -----------------------------------------------------------------
    pixel_t* dog = temp ;
    tempIsGrad = false ;
    {
      pixel_t* pt = dog ;
      for(int s = smin ; s <= smax-1 ; ++s) {
        pixel_t* srca = getLevel(o, s  ) ;
        pixel_t* srcb = getLevel(o, s+1) ;
        pixel_t* enda = srcb ;
        while( srca != enda ) {
          *pt++ = *srcb++ - *srca++ ;
        }
      }
    }

    // -----------------------------------------------------------------
    //                                           Find points of extremum
    // -----------------------------------------------------------------
    {
      pixel_t* pt  = dog + xo + yo + so ;
      for(int s = smin+1 ; s <= smax-2 ; ++s) {
        for(int y = 1 ; y < oh - 1 ; ++y) {
          for(int x = 1 ; x < ow - 1 ; ++x) {
            pixel_t v = *pt ;

            // assert( (pt - x*xo - y*yo - (s-smin)*so) - dog == 0 ) ;

#define CHECK_NEIGHBORS(CMP,SGN)                    \
            ( v CMP ## = SGN 0.8 * threshold &&     \
              v CMP *(pt + xo) &&                   \
              v CMP *(pt - xo) &&                   \
              v CMP *(pt + so) &&                   \
              v CMP *(pt - so) &&                   \
              v CMP *(pt + yo) &&                   \
              v CMP *(pt - yo) &&                   \
                                                    \
              v CMP *(pt + yo + xo) &&              \
              v CMP *(pt + yo - xo) &&              \
              v CMP *(pt - yo + xo) &&              \
              v CMP *(pt - yo - xo) &&              \
                                                    \
              v CMP *(pt + xo      + so) &&         \
              v CMP *(pt - xo      + so) &&         \
              v CMP *(pt + yo      + so) &&         \
              v CMP *(pt - yo      + so) &&         \
              v CMP *(pt + yo + xo + so) &&         \
              v CMP *(pt + yo - xo + so) &&         \
              v CMP *(pt - yo + xo + so) &&         \
              v CMP *(pt - yo - xo + so) &&         \
                                                    \
              v CMP *(pt + xo      - so) &&         \
              v CMP *(pt - xo      - so) &&         \
              v CMP *(pt + yo      - so) &&         \
              v CMP *(pt - yo      - so) &&         \
              v CMP *(pt + yo + xo - so) &&         \
              v CMP *(pt + yo - xo - so) &&         \
              v CMP *(pt - yo + xo - so) &&         \
              v CMP *(pt - yo - xo - so) )

            if( CHECK_NEIGHBORS(>,+) || CHECK_NEIGHBORS(<,-) ) {

              Keypoint k ;
              k.ix = x ;
              k.iy = y ;
              k.is = s ;
              keypoints.push_back(k) ;
            }
            pt += 1 ;
          }
          pt += 2 ;
        }
        pt += 2*yo ;
      }
    }

    // -----------------------------------------------------------------
    //                                               Refine local maxima
    // -----------------------------------------------------------------
    { // refine
      KeypointsIter siter ;
      KeypointsIter diter ;

      for(diter = siter = keypointsBegin() + nValidatedKeypoints ;
          siter != keypointsEnd() ;
          ++siter) {

        int x = int( siter->ix ) ;
        int y = int( siter->iy ) ;
        int s = int( siter->is ) ;

        VL::float_t Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0 ;
        VL::float_t  b [3] ;
        pixel_t* pt = 0;
        int dx = 0 ;
        int dy = 0 ;

        // must be exec. at least once
        for(int iter = 0 ; iter < 5 ; ++iter) {

          VL::float_t A[3*3] ;

          x += dx ;
          y += dy ;

          pt = dog
            + xo * x
            + yo * y
            + so * (s - smin) ;

#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))
#define Aat(i,j)     (A[(i)+(j)*3])

          /* Compute the gradient. */
          Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
          Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
          Ds = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;

          /* Compute the Hessian. */
          Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
          Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
          Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;

          Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) ) ;
          Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) ) ;
          Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) ) ;

          /* Solve linear system. */
          Aat(0,0) = Dxx ;
          Aat(1,1) = Dyy ;
          Aat(2,2) = Dss ;
          Aat(0,1) = Aat(1,0) = Dxy ;
          Aat(0,2) = Aat(2,0) = Dxs ;
          Aat(1,2) = Aat(2,1) = Dys ;

          b[0] = - Dx ;
          b[1] = - Dy ;
          b[2] = - Ds ;

          // Gauss elimination
          for(int j = 0 ; j < 3 ; ++j) {

            // look for leading pivot
            VL::float_t maxa = 0 ;
            VL::float_t maxabsa = 0 ;
            int   maxi = -1 ;
            int i ;
            for(i = j ; i < 3 ; ++i) {
              VL::float_t a    = Aat(i,j) ;
              VL::float_t absa = fabsf( a ) ;
              if ( absa > maxabsa ) {
                maxa    = a ;
                maxabsa = absa ;
                maxi    = i ;
              }
            }

            // singular?
            if( maxabsa < 1e-10f ) {
              b[0] = 0 ;
              b[1] = 0 ;
              b[2] = 0 ;
              break ;
            }

            i = maxi ;

            // swap j-th row with i-th row and
            // normalize j-th row
            for(int jj = j ; jj < 3 ; ++jj) {
              std::swap( Aat(j,jj) , Aat(i,jj) ) ;
              Aat(j,jj) /= maxa ;
            }
            std::swap( b[j], b[i] ) ;
            b[j] /= maxa ;

            // elimination
            for(int ii = j+1 ; ii < 3 ; ++ii) {
              VL::float_t x = Aat(ii,j) ;
              for(int jj = j ; jj < 3 ; ++jj) {
                Aat(ii,jj) -= x * Aat(j,jj) ;
              }
              b[ii] -= x * b[j] ;
            }
          }

          // backward substitution
          for(int i = 2 ; i > 0 ; --i) {
            VL::float_t x = b[i] ;
            for(int ii = i-1 ; ii >= 0 ; --ii) {
              b[ii] -= x * Aat(ii,i) ;
            }
          }

          /* If the translation of the keypoint is big, move the keypoint
           * and re-iterate the computation. Otherwise we are all set.
           */
          dx= ((b[0] >  0.6 && x < ow-2) ?  1 : 0 )
            + ((b[0] < -0.6 && x > 1   ) ? -1 : 0 ) ;

          dy= ((b[1] >  0.6 && y < oh-2) ?  1 : 0 )
            + ((b[1] < -0.6 && y > 1   ) ? -1 : 0 ) ;

          /*
          std::cout<<x<<","<<y<<"="<<at(0,0,0)
                   <<"("
                   <<at(0,0,0)+0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2])<<")"
                   <<" "<<std::flush ;
          */

          if( dx == 0 && dy == 0 ) break ;
        }

        /* std::cout<<std::endl ; */

        // Accept-reject keypoint
        {
          VL::float_t val = at(0,0,0) + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]) ;
          VL::float_t score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
          VL::float_t xn = x + b[0] ;
          VL::float_t yn = y + b[1] ;
          VL::float_t sn = s + b[2] ;

          if(fast_abs(val) > threshold &&
             score < (edgeThreshold+1)*(edgeThreshold+1)/edgeThreshold &&
             score >= 0 &&
             fast_abs(b[0]) < 1.5 &&
             fast_abs(b[1]) < 1.5 &&
             fast_abs(b[2]) < 1.5 &&
             xn >= 0    &&
             xn <= ow-1 &&
             yn >= 0    &&
             yn <= oh-1 &&
             sn >= smin &&
             sn <= smax ) {

            diter->o  = o ;

            diter->ix = x ;
            diter->iy = y ;
            diter->is = s ;

            diter->x = xn * xperiod ;
            diter->y = yn * xperiod ;
            diter->s = sn ;

            diter->sigma = getScaleFromIndex(o,sn) ;

            ++diter ;
          }
        }
      } // next candidate keypoint

      // prepare for next octave
      keypoints.resize( diter - keypoints.begin() ) ;
      nValidatedKeypoints = keypoints.size() ;
    } // refine block

  } // next octave
}

// ===================================================================
//                                       computeKeypointOrientations()
// -------------------------------------------------------------------

/** @brief Compute modulus and phase of the gradient
 **
 ** The function computes the modulus and the angle of the gradient of
 ** the specified octave @a o. The result is stored in a temporary
 ** internal buffer accessed by computeKeypointDescriptor() and
 ** computeKeypointOrientations().
 **
 ** The SIFT detector provides keypoint with scale index s in the
 ** range @c smin+1 and @c smax-2. As such, the buffer contains only
 ** these levels.
 **
 ** If called mutliple time on the same data, the function exits
 ** immediately.
 **
 ** @param o octave of interest.
 **/
void
Sift::prepareGrad(int o)
{
  int const ow = getOctaveWidth(o) ;
  int const oh = getOctaveHeight(o) ;
  int const xo = 1 ;
  int const yo = ow ;
  int const so = oh*ow ;

  if( ! tempIsGrad || tempOctave != o ) {

    // compute dx/dy
    for(int s = smin+1 ; s <= smax-2 ; ++s) {
      for(int y = 1 ; y < oh-1 ; ++y ) {
        pixel_t* src  = getLevel(o, s) + xo + yo*y ;
        pixel_t* end  = src + ow - 1 ;
        pixel_t* grad = 2 * (xo + yo*y + (s - smin -1)*so) + temp ;
        while(src != end) {
          VL::float_t Gx = 0.5 * ( *(src+xo) - *(src-xo) ) ;
          VL::float_t Gy = 0.5 * ( *(src+yo) - *(src-yo) ) ;
          VL::float_t m = fast_sqrt( Gx*Gx + Gy*Gy ) ;
          VL::float_t t = fast_mod_2pi( fast_atan2(Gy, Gx) + VL::float_t(2*CV_PI) );
          *grad++ = pixel_t( m ) ;
          *grad++ = pixel_t( t ) ;
          ++src ;
        }
      }
    }
  }

  tempIsGrad = true ;
  tempOctave = o ;
}

/** @brief Compute the orientation(s) of a keypoint
 **
 ** The function computes the orientation of the specified keypoint.
 ** The function returns up to four different orientations, obtained
 ** as strong peaks of the histogram of gradient orientations (a
 ** keypoint can theoretically generate more than four orientations,
 ** but this is very unlikely).
 **
 ** @remark The function needs to compute the gradient modululs and
 ** orientation of the Gaussian scale space octave to which the
 ** keypoint belongs. The result is cached, but discarded if different
 ** octaves are visited. Thereofre it is much quicker to evaluate the
 ** keypoints in their natural octave order.
 **
 ** The keypoint must lie within the scale space. In particular, the
 ** scale index is supposed to be in the range @c smin+1 and @c smax-1
 ** (this is from the SIFT detector). If this is not the case, the
 ** computation is silently aborted and no orientations are returned.
 **
 ** @param angles buffers to store the resulting angles.
 ** @param keypoint keypoint to process.
 ** @return number of orientations found.
 **/
int
Sift::computeKeypointOrientations(VL::float_t angles [4], Keypoint keypoint)
{
  int const   nbins = 36 ;
  VL::float_t const winFactor = 1.5 ;
  VL::float_t hist [nbins] ;

  // octave
  int o = keypoint.o ;
  VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

  // offsets to move in the Gaussian scale space octave
  const int ow = getOctaveWidth(o) ;
  const int oh = getOctaveHeight(o) ;
  const int xo = 2 ;
  const int yo = xo * ow ;
  const int so = yo * oh ;

  // keypoint fractional geometry
  VL::float_t x     = keypoint.x / xperiod ;
  VL::float_t y     = keypoint.y / xperiod ;
  VL::float_t sigma = keypoint.sigma / xperiod ;

  // shall we use keypoints.ix,iy,is here?
  int xi = ((int) (x+0.5)) ;
  int yi = ((int) (y+0.5)) ;
  int si = keypoint.is ;

  VL::float_t const sigmaw = winFactor * sigma ;
  int W = (int) floor(3.0 * sigmaw) ;

  // skip the keypoint if it is out of bounds
  if(o  < omin   ||
     o  >=omin+O ||
     xi < 0      ||
     xi > ow-1   ||
     yi < 0      ||
     yi > oh-1   ||
     si < smin+1 ||
     si > smax-2 ) {
    std::cerr<<"!"<<std::endl ;
    return 0 ;
  }

  // make sure that the gradient buffer is filled with octave o
  prepareGrad(o) ;

  // clear the SIFT histogram
  std::fill(hist, hist + nbins, 0.f) ;

  // fill the SIFT histogram
  pixel_t* pt = temp + xi * xo + yi * yo + (si - smin -1) * so ;

#undef at
#define at(dx,dy) (*(pt + (dx)*xo + (dy)*yo))

  for(int ys = std::max(-W, 1-yi) ; ys <= std::min(+W, oh -2 -yi) ; ++ys) {
    for(int xs = std::max(-W, 1-xi) ; xs <= std::min(+W, ow -2 -xi) ; ++xs) {

      VL::float_t dx = xi + xs - x;
      VL::float_t dy = yi + ys - y;
      VL::float_t r2 = dx*dx + dy*dy ;

      // limit to a circular window
      if(r2 >= W*W+0.5) continue ;

      VL::float_t wgt = VL::fast_expn( r2 / (2*sigmaw*sigmaw) ) ;
      VL::float_t mod = *(pt + xs*xo + ys*yo) ;
      VL::float_t ang = *(pt + xs*xo + ys*yo + 1) ;

      //      int bin = (int) floor( nbins * ang / (2*CV_PI) ) ;
      int bin = (int) floor( nbins * ang / (2*CV_PI) ) ;
      hist[bin] += mod * wgt ;
    }
  }

  // smooth the histogram
#if defined VL_LOWE_STRICT
  // Lowe's version apparently has a little issue with orientations
  // around + or - pi, which we reproduce here for compatibility
  for (int iter = 0; iter < 6; iter++) {
    VL::float_t prev  = hist[nbins/2] ;
    for (int i = nbins/2-1; i >= -nbins/2 ; --i) {
      int const j  = (i     + nbins) % nbins ;
      int const jp = (i - 1 + nbins) % nbins ;
      VL::float_t newh = (prev + hist[j] + hist[jp]) / 3.0;
      prev = hist[j] ;
      hist[j] = newh ;
    }
  }
#else
  // this is slightly more correct
  for (int iter = 0; iter < 6; iter++) {
    VL::float_t prev  = hist[nbins-1] ;
    VL::float_t first = hist[0] ;
    int i ;
    for (i = 0; i < nbins - 1; i++) {
      VL::float_t newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      prev = hist[i] ;
      hist[i] = newh ;
    }
    hist[i] = (prev + hist[i] + first)/3.0 ;
  }
#endif

  // find the histogram maximum
  VL::float_t maxh = * std::max_element(hist, hist + nbins) ;

  // find peaks within 80% from max
  int nangles = 0 ;
  for(int i = 0 ; i < nbins ; ++i) {
    VL::float_t h0 = hist [i] ;
    VL::float_t hm = hist [(i-1+nbins) % nbins] ;
    VL::float_t hp = hist [(i+1+nbins) % nbins] ;

    // is this a peak?
    if( h0 > 0.8*maxh && h0 > hm && h0 > hp ) {

      // quadratic interpolation
      //      VL::float_t di = -0.5 * (hp - hm) / (hp+hm-2*h0) ;
      VL::float_t di = -0.5 * (hp - hm) / (hp+hm-2*h0) ;
      VL::float_t th = 2*CV_PI * (i+di+0.5) / nbins ;
      angles [ nangles++ ] = th ;
      if( nangles == 4 )
        goto enough_angles ;
    }
  }
 enough_angles:
  return nangles ;
}

// ===================================================================
//                                         computeKeypointDescriptor()
// -------------------------------------------------------------------

namespace Detail {

/** Normalizes in norm L_2 a descriptor. */
void
normalize_histogram(VL::float_t* L_begin, VL::float_t* L_end)
{
  VL::float_t* L_iter ;
  VL::float_t norm = 0.0 ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    norm += (*L_iter) * (*L_iter) ;

  norm = fast_sqrt(norm) ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    *L_iter /= (norm + std::numeric_limits<VL::float_t>::epsilon() ) ;
}

}

/** @brief SIFT descriptor
 **
 ** The function computes the descriptor of the keypoint @a keypoint.
 ** The function fills the buffer @a descr_pt which must be large
 ** enough. The funciton uses @a angle0 as rotation of the keypoint.
 ** By calling the function multiple times, different orientations can
 ** be evaluated.
 **
 ** @remark The function needs to compute the gradient modululs and
 ** orientation of the Gaussian scale space octave to which the
 ** keypoint belongs. The result is cached, but discarded if different
 ** octaves are visited. Thereofre it is much quicker to evaluate the
 ** keypoints in their natural octave order.
 **
 ** The function silently abort the computations of keypoints without
 ** the scale space boundaries. See also siftComputeOrientations().
 **/
void
Sift::computeKeypointDescriptor
(VL::float_t* descr_pt,
 Keypoint keypoint,
 VL::float_t angle0)
{

  /* The SIFT descriptor is a  three dimensional histogram of the position
   * and orientation of the gradient.  There are NBP bins for each spatial
   * dimesions and NBO  bins for the orientation dimesion,  for a total of
   * NBP x NBP x NBO bins.
   *
   * The support  of each  spatial bin  has an extension  of SBP  = 3sigma
   * pixels, where sigma is the scale  of the keypoint.  Thus all the bins
   * together have a  support SBP x NBP pixels wide  . Since weighting and
   * interpolation of  pixel is used, another  half bin is  needed at both
   * ends of  the extension. Therefore, we  need a square window  of SBP x
   * (NBP + 1) pixels. Finally, since the patch can be arbitrarly rotated,
   * we need to consider  a window 2W += sqrt(2) x SBP  x (NBP + 1) pixels
   * wide.
   */

  // octave
  int o = keypoint.o ;
  VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

  // offsets to move in Gaussian scale space octave
  const int ow = getOctaveWidth(o) ;
  const int oh = getOctaveHeight(o) ;
  const int xo = 2 ;
  const int yo = xo * ow ;
  const int so = yo * oh ;

  // keypoint fractional geometry
  VL::float_t x     = keypoint.x / xperiod;
  VL::float_t y     = keypoint.y / xperiod ;
  VL::float_t sigma = keypoint.sigma / xperiod ;

  VL::float_t st0   = sinf( angle0 ) ;
  VL::float_t ct0   = cosf( angle0 ) ;

  // shall we use keypoints.ix,iy,is here?
  int xi = ((int) (x+0.5)) ;
  int yi = ((int) (y+0.5)) ;
  int si = keypoint.is ;

  // const VL::float_t magnif = 3.0f ;
  const int NBO = 8 ;
  const int NBP = 4 ;
  const VL::float_t SBP = magnif * sigma ;
  const int   W = (int) floor (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  /* Offsets to move in the descriptor. */
  /* Use Lowe's convention. */
  const int binto = 1 ;
  const int binyo = NBO * NBP ;
  const int binxo = NBO ;
  // const int bino  = NBO * NBP * NBP ;

  int bin ;

  // check bounds
  if(o  < omin   ||
     o  >=omin+O ||
     xi < 0      ||
     xi > ow-1   ||
     yi < 0      ||
     yi > oh-1   ||
     si < smin+1 ||
     si > smax-2 )
        return ;

  // make sure gradient buffer is up-to-date
  prepareGrad(o) ;

  std::fill( descr_pt, descr_pt + NBO*NBP*NBP, 0.f ) ;

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pixel_t const * pt = temp + xi*xo + yi*yo + (si - smin - 1)*so ;
  VL::float_t *  dpt = descr_pt + (NBP/2) * binyo + (NBP/2) * binxo ;

#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(int dyi = std::max(-W, 1-yi) ; dyi <= std::min(+W, oh-2-yi) ; ++dyi) {
    for(int dxi = std::max(-W, 1-xi) ; dxi <= std::min(+W, ow-2-xi) ; ++dxi) {

      // retrieve
      VL::float_t mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      VL::float_t angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      VL::float_t theta = fast_mod_2pi(-angle + angle0) ; // lowe compatible ?

      // fractional displacement
      VL::float_t dx = xi + dxi - x;
      VL::float_t dy = yi + dyi - y;

      // get the displacement normalized w.r.t. the keypoint
      // orientation and extension.
      VL::float_t nx = ( ct0 * dx + st0 * dy) / SBP ;
      VL::float_t ny = (-st0 * dx + ct0 * dy) / SBP ;
      VL::float_t nt = NBO * theta / (2*CV_PI) ;

      // Get the gaussian weight of the sample. The gaussian window
      // has a standard deviation equal to NBP/2. Note that dx and dy
      // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
      VL::float_t const wsigma = NBP/2 ;
      VL::float_t win = VL::fast_expn((nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

      // The sample will be distributed in 8 adjacent bins.
      // We start from the ``lower-left'' bin.
      int binx = fast_floor( nx - 0.5 ) ;
      int biny = fast_floor( ny - 0.5 ) ;
      int bint = fast_floor( nt ) ;
      VL::float_t rbinx = nx - (binx+0.5) ;
      VL::float_t rbiny = ny - (biny+0.5) ;
      VL::float_t rbint = nt - bint ;
      int dbinx ;
      int dbiny ;
      int dbint ;

      // Distribute the current sample into the 8 adjacent bins
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if( binx+dbinx >= -(NBP/2) &&
                binx+dbinx <   (NBP/2) &&
                biny+dbiny >= -(NBP/2) &&
                biny+dbiny <   (NBP/2) ) {
              VL::float_t weight = win
                * mod
                * fast_abs (1 - dbinx - rbinx)
                * fast_abs (1 - dbiny - rbiny)
                * fast_abs (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if( normalizeDescriptor ) {

    /* Normalize the histogram to L2 unit length. */
    Detail::normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;

    /* Truncate at 0.2. */
    for(bin = 0; bin < NBO*NBP*NBP ; ++bin) {
      if (descr_pt[bin] > 0.2) descr_pt[bin] = 0.2;
    }

    /* Normalize again. */
    Detail::normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
  }

}
// namespace VL
}

/****************************************************************************************\
  2.) wrapper of Vedaldi`s SIFT
\****************************************************************************************/

using namespace cv;

SIFT::CommonParams::CommonParams() :
        nOctaves(DEFAULT_NOCTAVES), nOctaveLayers(DEFAULT_NOCTAVE_LAYERS),
        firstOctave(DEFAULT_FIRST_OCTAVE), angleMode(FIRST_ANGLE)
{}

SIFT::CommonParams::CommonParams( int _nOctaves, int _nOctaveLayers, int _firstOctave, int _angleMode ) :
        nOctaves(_nOctaves), nOctaveLayers(_nOctaveLayers),
        firstOctave(_firstOctave), angleMode(_angleMode)
{}

SIFT::DetectorParams::DetectorParams() :
        threshold(GET_DEFAULT_THRESHOLD()), edgeThreshold(GET_DEFAULT_EDGE_THRESHOLD())
{}

SIFT::DetectorParams::DetectorParams( double _threshold, double _edgeThreshold ) :
        threshold(_threshold), edgeThreshold(_edgeThreshold)
{}

SIFT::DescriptorParams::DescriptorParams() :
        magnification(GET_DEFAULT_MAGNIFICATION()), isNormalize(DEFAULT_IS_NORMALIZE),
        recalculateAngles(true)
{}

SIFT::DescriptorParams::DescriptorParams( double _magnification, bool _isNormalize, bool _recalculateAngles ) :
        magnification(_magnification), isNormalize(_isNormalize),
        recalculateAngles(_recalculateAngles)
{}

SIFT::SIFT()
{}

SIFT::SIFT( double _threshold, double _edgeThreshold, int _nOctaves,
            int _nOctaveLayers, int _firstOctave, int _angleMode )
{
    detectorParams = DetectorParams(_threshold, _edgeThreshold);
    commParams = CommonParams(_nOctaves, _nOctaveLayers, _firstOctave, _angleMode);
}

SIFT::SIFT( double _magnification, bool _isNormalize, bool _recalculateAngles, int _nOctaves,
            int _nOctaveLayers, int _firstOctave, int _angleMode )
{
    descriptorParams = DescriptorParams(_magnification, _isNormalize, _recalculateAngles);
    commParams = CommonParams(_nOctaves, _nOctaveLayers, _firstOctave, _angleMode);
}

SIFT::SIFT( const CommonParams& _commParams,
            const DetectorParams& _detectorParams,
            const DescriptorParams& _descriptorParams )
{
    commParams = _commParams;
    detectorParams = _detectorParams;
    descriptorParams = _descriptorParams;
}

int SIFT::descriptorSize() const
{
    return DescriptorParams::DESCRIPTOR_SIZE;
}

SIFT::CommonParams SIFT::getCommonParams () const
{
    return commParams;
}

SIFT::DetectorParams SIFT::getDetectorParams () const
{
    return detectorParams;
}

SIFT::DescriptorParams SIFT::getDescriptorParams () const
{
    return descriptorParams;
}

inline KeyPoint vlKeypointToOcv( const VL::Sift& vlSift, const VL::Sift::Keypoint& vlKeypoint, float angle )
{
    float size = vlKeypoint.sigma*SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION()*4;// 4==NBP
    return KeyPoint( vlKeypoint.x, vlKeypoint.y, size, angle, 0, vlKeypoint.o, 0 );
}

inline void ocvKeypointToVl( const VL::Sift& vlSift, const KeyPoint& ocvKeypoint,
                             VL::Sift::Keypoint& vlKeypoint, int magnification )
{
    float sigma = ocvKeypoint.size/(SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION()*4);// 4==NBP
    vlKeypoint = vlSift.getKeypoint( ocvKeypoint.pt.x, ocvKeypoint.pt.y, sigma);
}

bool computeKeypointOrientations( VL::Sift& sift, const VL::Sift::Keypoint& keypoint, float& angleVal, int angleMode )
{
    angleVal = 0.f;
    VL::float_t angles[4];

    int angleCount = sift.computeKeypointOrientations(angles, keypoint);
    if( angleCount > 0 )
    {
        if( angleMode == SIFT::CommonParams::FIRST_ANGLE )
        {
            angleVal = angles[0];
        }
        else if( angleMode == SIFT::CommonParams::AVERAGE_ANGLE )
        {
            for( int i = 0; i < angleCount; i++ )
                angleVal += angles[i];
            angleVal /= angleCount;
        }
        else
        {
          assert(0);
        }
        return true;
    }

    return false;
}


struct KeyPoint_LessThan
{
    KeyPoint_LessThan(const vector<KeyPoint>& _kp) : kp(&_kp) {}
    bool operator()(int i, int j) const
    {
        const KeyPoint& kp1 = (*kp)[i];
        const KeyPoint& kp2 = (*kp)[j];
        if( kp1.pt.x != kp2.pt.x )
            return kp1.pt.x < kp2.pt.x;
        if( kp1.pt.y != kp2.pt.y )
            return kp1.pt.y < kp2.pt.y;
        if( kp1.size != kp2.size )
            return kp1.size > kp2.size;
        if( kp1.size != kp2.size )
            return kp1.size > kp2.size;
        if( kp1.angle != kp2.angle )
            return kp1.angle < kp2.angle;
        if( kp1.response != kp2.response )
            return kp1.response > kp2.response;
        if( kp1.octave != kp2.octave )
            return kp1.octave > kp2.octave;
        if( kp1.class_id != kp2.class_id )
            return kp1.class_id > kp2.class_id;
        
        return i < j;
    }
    const vector<KeyPoint>* kp;
};


static void removeDuplicatedKeypoints(vector<KeyPoint>& keypoints)
{
    int i, j, n = (int)keypoints.size();
    vector<int> kpidx(n);
    vector<uchar> mask(n, (uchar)1);
    
    for( i = 0; i < n; i++ )
        kpidx[i] = i;
    std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
    for( i = 1, j = 0; i < n; i++ )
    {
        KeyPoint& kp1 = keypoints[kpidx[i]];
        KeyPoint& kp2 = keypoints[kpidx[j]];
        if( kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y ||
            kp1.size != kp2.size || kp1.angle != kp2.angle )
            j = i;
        else
            mask[kpidx[i]] = 0;
    }

    for( i = j = 0; i < n; i++ )
    {
        if( mask[i] )
        {
            if( i != j )
                keypoints[j] = keypoints[i];
            j++;
        }
    }
    keypoints.resize(j);
}


// detectors
void SIFT::operator()(const Mat& img, const Mat& mask,
                      vector<KeyPoint>& keypoints) const
{
    if( img.empty() || img.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "img is empty or has incorrect type" );

    Mat fimg;
    img.convertTo(fimg, CV_32FC1, 1.0/255.0);

    const double sigman = .5 ;
    const double sigma0 = 1.6 * powf(2.0f, 1.0f / commParams.nOctaveLayers) ;
    const double a_180divPI = 180./CV_PI;

    VL::Sift vlsift((float*)fimg.data, fimg.cols, fimg.rows,
                    sigman, sigma0, commParams.nOctaves, commParams.nOctaveLayers,
                    commParams.firstOctave, -1, commParams.nOctaveLayers+1);

    vlsift.detectKeypoints(detectorParams.threshold, detectorParams.edgeThreshold);
    int d = std::abs(int(vlsift.keypointsBegin()-vlsift.keypointsEnd()));
    keypoints.reserve(d);

    for( VL::Sift::KeypointsConstIter iter = vlsift.keypointsBegin(); iter != vlsift.keypointsEnd(); ++iter )
    {
        float angleVal = 0.f;
        if( computeKeypointOrientations( vlsift, *iter, angleVal, commParams.angleMode ) )
        {
            keypoints.push_back( vlKeypointToOcv(vlsift, *iter, angleVal*a_180divPI) );
        }
    }
    removeDuplicatedKeypoints(keypoints);
}

struct InvalidKeypoint
{
    bool operator()(const KeyPoint& kp) const { return kp.octave == std::numeric_limits<int>::max(); }
};

// descriptors
void SIFT::operator()(const Mat& img, const Mat& mask,
                      vector<KeyPoint>& keypoints,
                      Mat& descriptors,
                      bool useProvidedKeypoints) const
{
    if( img.empty() || img.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "img is empty or has incorrect type" );

    Mat fimg;
    img.convertTo(fimg, CV_32FC1, 1.0/255.0);

    const double sigman = .5 ;
    const double sigma0 = 1.6 * powf(2.0f, 1.0f / commParams.nOctaveLayers) ;
    const double a_180divPI = 180./CV_PI;
    const double a_PIdiv180 = CV_PI/180.;

    if( !useProvidedKeypoints )
        (*this)(img, mask, keypoints);

    VL::Sift vlsift((float*)fimg.data, fimg.cols, fimg.rows,
                    sigman, sigma0, commParams.nOctaves, commParams.nOctaveLayers,
                    commParams.firstOctave, -1, commParams.nOctaveLayers+1);
    vlsift.setNormalizeDescriptor(descriptorParams.isNormalize);
    vlsift.setMagnification(descriptorParams.magnification);

    descriptors.create( keypoints.size(), DescriptorParams::DESCRIPTOR_SIZE, DataType<VL::float_t>::type );
    vector<KeyPoint>::iterator kp_iter = keypoints.begin();

    for( int pi = 0 ; kp_iter != keypoints.end(); ++kp_iter, pi++ )
    {
        VL::Sift::Keypoint vlkpt;
        ocvKeypointToVl( vlsift, *kp_iter, vlkpt, descriptorParams.magnification );

        if( descriptorParams.recalculateAngles )
        {
            float recalcAngleVal = 0.f;
            if( computeKeypointOrientations( vlsift, vlkpt, recalcAngleVal, commParams.angleMode ) )
            {
                kp_iter->angle = recalcAngleVal*a_180divPI; // save recalculated angle value
                assert( kp_iter->angle >= 0 );
                vlsift.computeKeypointDescriptor((VL::float_t*)descriptors.ptr(pi), vlkpt, recalcAngleVal);
            }
            else
            {
                // mark point to remove
                kp_iter->octave = std::numeric_limits<int>::max();
            }
        }
        else
        {
            if( kp_iter->angle < 0 )
                CV_Error( CV_StsBadArg, "Angle must be applicable (i.e. supported by feature detector that was used to detect keypoints)." );

            float angleVal = kp_iter->angle*a_PIdiv180;
            vlsift.computeKeypointDescriptor((VL::float_t*)descriptors.ptr(pi), vlkpt, angleVal);
        }
    }

    if( descriptorParams.recalculateAngles )
        keypoints.erase( remove_if(keypoints.begin(), keypoints.end(), InvalidKeypoint()), keypoints.end());
}

#endif // ARM_NO_SIFT
