// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef OPENCV_3D_UTILS_HPP
#define OPENCV_3D_UTILS_HPP

#include "../precomp.hpp"

namespace cv
{

 /** Checks if the value is a valid depth. For CV_16U or CV_16S, the convention is to be invalid if it is
  * a limit. For a float/double, we just check if it is a NaN
  * @param depth the depth to check for validity
  */
inline bool isValidDepth(const float& depth)
{
    return !cvIsNaN(depth);
}

inline bool isValidDepth(const double& depth)
{
    return !cvIsNaN(depth);
}

inline bool isValidDepth(const short int& depth)
{
    return (depth != std::numeric_limits<short int>::min()) &&
           (depth != std::numeric_limits<short int>::max());
}

inline bool isValidDepth(const unsigned short int& depth)
{
    return (depth != std::numeric_limits<unsigned short int>::min()) &&
           (depth != std::numeric_limits<unsigned short int>::max());
}

inline bool isValidDepth(const int& depth)
{
    return (depth != std::numeric_limits<int>::min()) &&
           (depth != std::numeric_limits<int>::max());
}

inline bool isValidDepth(const unsigned int& depth)
{
    return (depth != std::numeric_limits<unsigned int>::min()) &&
           (depth != std::numeric_limits<unsigned int>::max());
}


// One place to turn intrinsics on and off
#define USE_INTRINSICS CV_SIMD128

typedef float depthType;

const float qnan = std::numeric_limits<float>::quiet_NaN();
const cv::Vec3f nan3(qnan, qnan, qnan);
#if USE_INTRINSICS
const cv::v_float32x4 nanv(qnan, qnan, qnan, qnan);
#endif

inline bool isNaN(cv::Point3f p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}

#if USE_INTRINSICS
static inline bool isNaN(const cv::v_float32x4& p)
{
    return cv::v_check_any(v_ne(p, p));
}
#endif

inline size_t roundDownPow2(size_t x)
{
    size_t shift = 0;
    while(x != 0)
    {
        shift++; x >>= 1;
    }
    return (size_t)(1ULL << (shift-1));
}

template<> class DataType<cv::Point3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 3,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<cv::Vec3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 3,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<cv::Vec4f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 4,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};


typedef cv::Vec4f ptype;
inline cv::Vec3f fromPtype(const ptype& x)
{
    return cv::Vec3f(x[0], x[1], x[2]);
}

inline ptype toPtype(const cv::Vec3f& x)
{
    return ptype(x[0], x[1], x[2], 0);
}

enum
{
    DEPTH_TYPE = DataType<depthType>::type,
    POINT_TYPE = DataType<ptype    >::type,
    COLOR_TYPE = DataType<ptype    >::type
};

typedef cv::Mat_< ptype > Points;
typedef Points Normals;
typedef Points Colors;

typedef cv::Point3f _ptype;
typedef cv::Mat_< _ptype > _Points;
typedef _Points _Normals;
typedef _Points _Colors;

enum
{
    _DEPTH_TYPE = DataType<depthType>::type,
    _POINT_TYPE = DataType<_ptype   >::type,
    _COLOR_TYPE = DataType<_ptype   >::type
};

typedef cv::Mat_< depthType > Depth;

void makeFrameFromDepth(InputArray depth, OutputArray pyrPoints, OutputArray pyrNormals,
                        const Matx33f intr, int levels, float depthFactor,
                        float sigmaDepth, float sigmaSpatial, int kernelSize,
                        float truncateThreshold);
void buildPyramidPointsNormals(InputArray _points, InputArray _normals,
                               OutputArrayOfArrays pyrPoints, OutputArrayOfArrays pyrNormals,
                               int levels);

struct Intr
{
    /** @brief Camera intrinsics */
    /** Reprojects screen point to camera space given z coord. */
    struct Reprojector
    {
        Reprojector() {}
        inline Reprojector(Intr intr)
        {
            fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;
            cx = intr.cx, cy = intr.cy;
        }
        template<typename T>
        inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
        {
            T x = p.z * (p.x - cx) * fxinv;
            T y = p.z * (p.y - cy) * fyinv;
            return cv::Point3_<T>(x, y, p.z);
        }

        float fxinv, fyinv, cx, cy;
    };

    /** Projects camera space vector onto screen */
    struct Projector
    {
        inline Projector(Intr intr) : fx(intr.fx), fy(intr.fy), cx(intr.cx), cy(intr.cy) { }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p) const
        {
            T invz = T(1)/p.z;
            T x = fx*(p.x*invz) + cx;
            T y = fy*(p.y*invz) + cy;
            return cv::Point_<T>(x, y);
        }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p, cv::Point3_<T>& pixVec) const
        {
            T invz = T(1)/p.z;
            pixVec = cv::Point3_<T>(p.x*invz, p.y*invz, 1);
            T x = fx*pixVec.x + cx;
            T y = fy*pixVec.y + cy;
            return cv::Point_<T>(x, y);
        }
        float fx, fy, cx, cy;
    };
    Intr() : fx(), fy(), cx(), cy() { }
    Intr(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy) { }
    Intr(cv::Matx33f m) : fx(m(0, 0)), fy(m(1, 1)), cx(m(0, 2)), cy(m(1, 2)) { }
    // scale intrinsics to pyramid level
    inline Intr scale(int pyr) const
    {
        float factor = (1.f /(1 << pyr));
        return Intr(fx*factor, fy*factor, cx*factor, cy*factor);
    }
    inline Reprojector makeReprojector() const { return Reprojector(*this); }
    inline Projector   makeProjector()   const { return Projector(*this);   }

    inline cv::Matx33f getMat() const { return Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1); }

    float fx, fy, cx, cy;
};

class OdometryFrame::Impl
{
public:
    Impl() : pyramids(OdometryFramePyramidType::N_PYRAMIDS) { }
    virtual ~Impl() {}

    virtual void getImage(OutputArray image) const ;
    virtual void getGrayImage(OutputArray image) const ;
    virtual void getDepth(OutputArray depth) const ;
    virtual void getProcessedDepth(OutputArray depth) const ;
    virtual void getMask(OutputArray mask) const ;
    virtual void getNormals(OutputArray normals) const ;

    virtual int getPyramidLevels() const ;

    virtual void getPyramidAt(OutputArray img,
                              OdometryFramePyramidType pyrType, size_t level) const ;

    UMat imageGray;
    UMat image;
    UMat depth;
    UMat scaledDepth;
    UMat mask;
    UMat normals;
    std::vector< std::vector<UMat> > pyramids;
};

} // namespace cv


#endif // include guard
