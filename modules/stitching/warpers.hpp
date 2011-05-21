#ifndef __OPENCV_WARPERS_HPP__
#define __OPENCV_WARPERS_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class Warper
{
public:
    enum { PLANE, CYLINDRICAL, SPHERICAL };
    static cv::Ptr<Warper> createByCameraFocal(float focal, int type);

    virtual ~Warper() {}
    virtual cv::Point warp(const cv::Mat &src, float focal, const cv::Mat& R, cv::Mat &dst,
                           int interp_mode = cv::INTER_LINEAR, int border_mode = cv::BORDER_REFLECT) = 0;
};


struct ProjectorBase
{
    void setTransformation(const cv::Mat& R);

    cv::Size size;
    float focal;
    float r[9];
    float rinv[9];
    float scale;
};


template <class P>
class WarperBase : public Warper
{   
public:
    cv::Point warp(const cv::Mat &src, float focal, const cv::Mat &R, cv::Mat &dst,
                   int interp_mode, int border_mode);

protected:
    // Detects ROI of the destination image. It's correct for any projection.
    virtual void detectResultRoi(cv::Point &dst_tl, cv::Point &dst_br);

    // Detects ROI of the destination image by walking over image border.
    // Correctness for any projection isn't guaranteed.
    void detectResultRoiByBorder(cv::Point &dst_tl, cv::Point &dst_br);

    cv::Size src_size_;
    P projector_;
};


struct PlaneProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);

    float plane_dist;
};


// Projects image onto z = plane_dist plane
class PlaneWarper : public WarperBase<PlaneProjector>
{
public:
    PlaneWarper(float plane_dist = 1.f, float scale = 1.f)
    {
        projector_.plane_dist = plane_dist;
        projector_.scale = scale;
    }

private:
    void detectResultRoi(cv::Point &dst_tl, cv::Point &dst_br);
};


struct SphericalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located at (0, -1, 0) and (0, 1, 0) points.
class SphericalWarper : public WarperBase<SphericalProjector>
{
public:
    SphericalWarper(float scale = 300.f) { projector_.scale = scale; }

private:  
    void detectResultRoi(cv::Point &dst_tl, cv::Point &dst_br);
};


struct CylindricalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto x * x + z * z = 1 cylinder
class CylindricalWarper : public WarperBase<CylindricalProjector>
{
public:
    CylindricalWarper(float scale = 300.f) { projector_.scale = scale; }

private:
    void detectResultRoi(cv::Point &dst_tl, cv::Point &dst_br)
    {
        WarperBase<CylindricalProjector>::detectResultRoiByBorder(dst_tl, dst_br);
    }
};

#include "warpers_inl.hpp"

#endif // __OPENCV_WARPERS_HPP__
