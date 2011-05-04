#include "warpers.hpp"

using namespace std;
using namespace cv;

Ptr<Warper> Warper::createByCameraFocal(int focal, int type)
{
    if (type == PLANE)
        return new PlaneWarper(focal);
    if (type == CYLINDRICAL)
        return new CylindricalWarper(focal);
    if (type == SPHERICAL)
        return new SphericalWarper(focal);
    CV_Error(CV_StsBadArg, "unsupported warping type");
    return NULL;
}


void ProjectorBase::setCameraMatrix(const Mat &M)
{
    CV_Assert(M.size() == Size(3, 3));
    CV_Assert(M.type() == CV_32F);
    m[0] = M.at<float>(0, 0); m[1] = M.at<float>(0, 1); m[2] = M.at<float>(0, 2);
    m[3] = M.at<float>(1, 0); m[4] = M.at<float>(1, 1); m[5] = M.at<float>(1, 2);
    m[6] = M.at<float>(2, 0); m[7] = M.at<float>(2, 1); m[8] = M.at<float>(2, 2);

    Mat M_inv = M.inv();
    minv[0] = M_inv.at<float>(0, 0); minv[1] = M_inv.at<float>(0, 1); minv[2] = M_inv.at<float>(0, 2);
    minv[3] = M_inv.at<float>(1, 0); minv[4] = M_inv.at<float>(1, 1); minv[5] = M_inv.at<float>(1, 2);
    minv[6] = M_inv.at<float>(2, 0); minv[7] = M_inv.at<float>(2, 1); minv[8] = M_inv.at<float>(2, 2);
}


Point Warper::operator ()(const Mat &src, float focal, const Mat& M, Mat &dst,
                          int interp_mode, int border_mode)
{
    return warp(src, focal, M, dst, interp_mode, border_mode);
}


void PlaneWarper::detectResultRoi(Point &dst_tl, Point &dst_br)
{
    float tl_uf = numeric_limits<float>::max();
    float tl_vf = numeric_limits<float>::max();
    float br_uf = -numeric_limits<float>::max();
    float br_vf = -numeric_limits<float>::max();

    float u, v;

    projector_.mapForward(0, 0, u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(0, static_cast<float>(src_size_.height - 1), u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size_.width - 1), 0, u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size_.width - 1), static_cast<float>(src_size_.height - 1), u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


void SphericalWarper::detectResultRoi(Point &dst_tl, Point &dst_br)
{
    detectResultRoiByBorder(dst_tl, dst_br);

    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

    float x = projector_.minv[1];
    float y = projector_.minv[4];
    float z = projector_.minv[7];
    if (y > 0.f)
    {
        x = projector_.focal * x / z + src_size_.width * 0.5f;
        y = projector_.focal * y / z + src_size_.height * 0.5f;
        if (x > 0.f && x < src_size_.width && y > 0.f && y < src_size_.height)
        {
            tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
            br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(CV_PI * projector_.scale));
        }
    }

    x = projector_.minv[1];
    y = -projector_.minv[4];
    z = projector_.minv[7];
    if (y > 0.f)
    {
        x = projector_.focal * x / z + src_size_.width * 0.5f;
        y = projector_.focal * y / z + src_size_.height * 0.5f;
        if (x > 0.f && x < src_size_.width && y > 0.f && y < src_size_.height)
        {
            tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(0));
            br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}
