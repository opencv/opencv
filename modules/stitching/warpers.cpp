#include "warpers.hpp"

using namespace std;
using namespace cv;

Ptr<Warper> Warper::createByCameraFocal(float focal, int type)
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


void ProjectorBase::setTransformation(const Mat &R)
{
    CV_Assert(R.size() == Size(3, 3));
    CV_Assert(R.type() == CV_32F);
    r[0] = R.at<float>(0, 0); r[1] = R.at<float>(0, 1); r[2] = R.at<float>(0, 2);
    r[3] = R.at<float>(1, 0); r[4] = R.at<float>(1, 1); r[5] = R.at<float>(1, 2);
    r[6] = R.at<float>(2, 0); r[7] = R.at<float>(2, 1); r[8] = R.at<float>(2, 2);

    Mat Rinv = R.inv();
    rinv[0] = Rinv.at<float>(0, 0); rinv[1] = Rinv.at<float>(0, 1); rinv[2] = Rinv.at<float>(0, 2);
    rinv[3] = Rinv.at<float>(1, 0); rinv[4] = Rinv.at<float>(1, 1); rinv[5] = Rinv.at<float>(1, 2);
    rinv[6] = Rinv.at<float>(2, 0); rinv[7] = Rinv.at<float>(2, 1); rinv[8] = Rinv.at<float>(2, 2);
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

    float x = projector_.rinv[1];
    float y = projector_.rinv[4];
    float z = projector_.rinv[7];
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

    x = projector_.rinv[1];
    y = -projector_.rinv[4];
    z = projector_.rinv[7];
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
