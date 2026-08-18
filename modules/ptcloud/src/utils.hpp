// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_UTILS_H_
#define _CODERS_UTILS_H_

#include "precomp.hpp"

#include <opencv2/core/quaternion.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace cv {

std::vector<std::string> split(const std::string &s, char delimiter);

inline bool startsWith(const std::string &s1, const std::string &s2)
{
    return s1.compare(0, s2.length(), s2) == 0;
}

inline std::string trimSpaces(const std::string &input)
{
    size_t start = 0;
    while (start < input.size() && input[start] == ' ')
    {
        start++;
    }
    size_t end = input.size();
    while (end > start && (input[end - 1] == ' ' || input[end - 1] == '\n' || input[end - 1] == '\r'))
    {
        end--;
    }
    return input.substr(start, end - start);
}

inline std::string getExtension(const std::string& filename)
{
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos)
    {
        return "";
    }
    return filename.substr( pos + 1);
}

template <typename T>
void swapEndian(T &val)
{
    union U
    {
        T val;
        std::array<std::uint8_t, sizeof(T)> raw;
    } src, dst;

    src.val = val;
    std::reverse_copy(src.raw.begin(), src.raw.end(), dst.raw.begin());
    val = dst.val;
}

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

// 3D Gaussian Splatting attribute decoding and depth sorting. Deliberately free of
// OpenGL so it stays testable on headless builds.
namespace splat {

enum
{
    STRIDE = 13,
    OFS_POS = 0,
    OFS_COV = 3,
    OFS_RGB = 9,
    OFS_ALPHA = 12,

    // Column order of plyProperties(), which decode() reads.
    RAW_STRIDE = 14,
    RAW_OFS_POS = 0,
    RAW_OFS_DC = 3,
    RAW_OFS_OPACITY = 6,
    RAW_OFS_SCALE = 7,
    RAW_OFS_ROT = 10,

    // Byte layout of a ".splat" record, which decodePacked() reads.
    PACKED_STRIDE = 32,
    PACKED_OFS_POS = 0,
    PACKED_OFS_SCALE = 12,
    PACKED_OFS_RGBA = 24,
    PACKED_OFS_ROT = 28
};

inline const std::vector<std::string>& plyProperties()
{
    static const std::vector<std::string> names = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
    return names;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float shDcToColor(float dc)
{
    return std::min(1.0f, std::max(0.0f, 0.5f + 0.28209479177387814f * dc));
}

// Sigma = R S S^T R^T, symmetric positive semi-definite for any rotation and scale.
inline Matx33f covariance(const Vec3f& scale, const Vec4f& rot)
{
    Matx33f rs = Quatf(rot[0], rot[1], rot[2], rot[3]).toRotMat3x3(QUAT_ASSUME_NOT_UNIT);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rs(i, j) *= scale[j];
    return rs * rs.t();
}

inline void decode(const Mat& raw, Mat& splats)
{
    CV_Assert(raw.type() == CV_32F && raw.cols == RAW_STRIDE);

    splats.create(raw.rows, STRIDE, CV_32F);
    for (int i = 0; i < raw.rows; i++)
    {
        const float* s = raw.ptr<float>(i);
        float* d = splats.ptr<float>(i);

        for (int k = 0; k < 3; k++)
            d[OFS_POS + k] = s[RAW_OFS_POS + k];

        Vec3f scale(std::exp(s[RAW_OFS_SCALE + 0]),
                    std::exp(s[RAW_OFS_SCALE + 1]),
                    std::exp(s[RAW_OFS_SCALE + 2]));
        Matx33f cov = covariance(scale, Vec4f(s[RAW_OFS_ROT + 0], s[RAW_OFS_ROT + 1],
                                              s[RAW_OFS_ROT + 2], s[RAW_OFS_ROT + 3]));

        d[OFS_COV + 0] = cov(0, 0);
        d[OFS_COV + 1] = cov(0, 1);
        d[OFS_COV + 2] = cov(0, 2);
        d[OFS_COV + 3] = cov(1, 1);
        d[OFS_COV + 4] = cov(1, 2);
        d[OFS_COV + 5] = cov(2, 2);

        for (int k = 0; k < 3; k++)
            d[OFS_RGB + k] = shDcToColor(s[RAW_OFS_DC + k]);

        d[OFS_ALPHA] = sigmoid(s[RAW_OFS_OPACITY]);
    }
}

// Values arrive already activated, so only the covariance is built.
inline void decodePacked(const uchar* data, int n, Mat& splats)
{
    CV_Assert(data != nullptr && n >= 0);

    splats.create(n, STRIDE, CV_32F);
    for (int i = 0; i < n; i++)
    {
        const uchar* s = data + (size_t)i * PACKED_STRIDE;
        float* d = splats.ptr<float>(i);

        Vec3f pos, scale;
        memcpy(pos.val, s + PACKED_OFS_POS, sizeof(pos.val));
        memcpy(scale.val, s + PACKED_OFS_SCALE, sizeof(scale.val));

        for (int k = 0; k < 3; k++)
            d[OFS_POS + k] = pos[k];

        Vec4f rot;
        for (int k = 0; k < 4; k++)
            rot[k] = (s[PACKED_OFS_ROT + k] - 128.f) / 128.f;
        if (rot.dot(rot) < 1e-12f)
            rot = Vec4f(1.f, 0.f, 0.f, 0.f);

        Matx33f cov = covariance(scale, rot);

        d[OFS_COV + 0] = cov(0, 0);
        d[OFS_COV + 1] = cov(0, 1);
        d[OFS_COV + 2] = cov(0, 2);
        d[OFS_COV + 3] = cov(1, 1);
        d[OFS_COV + 4] = cov(1, 2);
        d[OFS_COV + 5] = cov(2, 2);

        for (int k = 0; k < 3; k++)
            d[OFS_RGB + k] = s[PACKED_OFS_RGBA + k] / 255.f;

        d[OFS_ALPHA] = s[PACKED_OFS_RGBA + 3] / 255.f;
    }
}

inline void sortByDepth(const Mat& splats, const Vec3f& cam, std::vector<int>& order)
{
    CV_Assert(splats.type() == CV_32F && splats.cols == STRIDE);

    const int n = splats.rows;
    std::vector<float> key(n);
    order.resize(n);
    for (int i = 0; i < n; i++)
    {
        const float* p = splats.ptr<float>(i);
        Vec3f d(p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]);
        key[i] = d.dot(d);
        order[i] = i;
    }

    std::sort(order.begin(), order.end(),
              [&key](int a, int b) { return key[a] > key[b]; });
}

} // namespace splat

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

} /* namespace cv */

#endif
