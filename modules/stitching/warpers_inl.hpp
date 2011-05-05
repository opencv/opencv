#ifndef __OPENCV_WARPERS_INL_HPP__
#define __OPENCV_WARPERS_INL_HPP__

#include "warpers.hpp" // Make your IDE see declarations

template <class P>
cv::Point WarperBase<P>::warp(const cv::Mat &src, float focal, const cv::Mat &M, cv::Mat &dst,
                              int interp_mode, int border_mode)
{
    src_size_ = src.size();

    projector_.size = src.size();
    projector_.focal = focal;
    projector_.setCameraMatrix(M);

    cv::Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    cv::Mat xmap(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
    cv::Mat ymap(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

    float x, y;
    for (int v = dst_tl.y; v <= dst_br.y; ++v)
    {
        for (int u = dst_tl.x; u <= dst_br.x; ++u)
        {
            projector_.mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
            xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
            ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
        }
    }

    dst.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, src.type());
    remap(src, dst, xmap, ymap, interp_mode, border_mode);

    return dst_tl;
}


template <class P>
void WarperBase<P>::detectResultRoi(cv::Point &dst_tl, cv::Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    float u, v;
    for (int y = 0; y < src_size_.height; ++y)
    {
        for (int x = 0; x < src_size_.width; ++x)
        {
            projector_.mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
            tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
            br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


template <class P>
void WarperBase<P>::detectResultRoiByBorder(cv::Point &dst_tl, cv::Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    float u, v;
    for (float x = 0; x < src_size_.width; ++x)
    {
        projector_.mapForward(static_cast<float>(x), 0, u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(x), static_cast<float>(src_size_.height - 1), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
    }
    for (int y = 0; y < src_size_.height; ++y)
    {
        projector_.mapForward(0, static_cast<float>(y), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(src_size_.width - 1), static_cast<float>(y), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


inline
void PlaneProjector::mapForward(float x, float y, float &u, float &v)
{
    x -= size.width * 0.5f;
    y -= size.height * 0.5f;

    float x_ = m[0] * x + m[1] * y + m[2] * focal;
    float y_ = m[3] * x + m[4] * y + m[5] * focal;
    float z_ = m[6] * x + m[7] * y + m[8] * focal;

    u = scale * x_ / z_ * plane_dist;
    v = scale * y_ / z_ * plane_dist;
}


inline
void PlaneProjector::mapBackward(float u, float v, float &x, float &y)
{
    float x_ = u / scale;
    float y_ = v / scale;

    float z;
    x = minv[0] * x_ + minv[1] * y_ + minv[2] * plane_dist;
    y = minv[3] * x_ + minv[4] * y_ + minv[5] * plane_dist;
    z = minv[6] * x_ + minv[7] * y_ + minv[8] * plane_dist;

    x = focal * x / z + size.width * 0.5f;
    y = focal * y / z + size.height * 0.5f;
}


inline
void SphericalProjector::mapForward(float x, float y, float &u, float &v)
{
    x -= size.width * 0.5f;
    y -= size.height * 0.5f;

    float x_ = m[0] * x + m[1] * y + m[2] * focal;
    float y_ = m[3] * x + m[4] * y + m[5] * focal;
    float z_ = m[6] * x + m[7] * y + m[8] * focal;

    u = scale * atan2f(x_, z_);
    v = scale * (static_cast<float>(CV_PI) - acosf(y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_)));
}


inline
void SphericalProjector::mapBackward(float u, float v, float &x, float &y)
{
    float sinv = sinf(static_cast<float>(CV_PI) - v / scale);
    float x_ = sinv * sinf(u / scale);
    float y_ = cosf(static_cast<float>(CV_PI) - v / scale);
    float z_ = sinv * cosf(u / scale);

    float z;
    x = minv[0] * x_ + minv[1] * y_ + minv[2] * z_;
    y = minv[3] * x_ + minv[4] * y_ + minv[5] * z_;
    z = minv[6] * x_ + minv[7] * y_ + minv[8] * z_;

    x = focal * x / z + size.width * 0.5f;
    y = focal * y / z + size.height * 0.5f;
}


inline
void CylindricalProjector::mapForward(float x, float y, float &u, float &v)
{
    x -= size.width * 0.5f;
    y -= size.height * 0.5f;

    float x_ = m[0] * x + m[1] * y + m[2] * focal;
    float y_ = m[3] * x + m[4] * y + m[5] * focal;
    float z_ = m[6] * x + m[7] * y + m[8] * focal;

    u = scale * atan2f(x_, z_);
    v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}


inline
void CylindricalProjector::mapBackward(float u, float v, float &x, float &y)
{
    float x_ = sinf(u / scale);
    float y_ = v / scale;
    float z_ = cosf(u / scale);

    float z;
    x = minv[0] * x_ + minv[1] * y_ + minv[2] * z_;
    y = minv[3] * x_ + minv[4] * y_ + minv[5] * z_;
    z = minv[6] * x_ + minv[7] * y_ + minv[8] * z_;

    x = focal * x / z + size.width * 0.5f;
    y = focal * y / z + size.height * 0.5f;
}

#endif // __OPENCV_WARPERS_INL_HPP__
