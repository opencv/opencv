#include "opencv2/stitching/detail/warpers.hpp"


namespace cv {

class CV_EXPORTS_W PyRotationWarper
{
    Ptr<detail::RotationWarper> rw;

public:
    CV_WRAP PyRotationWarper(String type, float scale);
    ~PyRotationWarper() {}

    /** @brief Projects the image point.

    @param pt Source point
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @return Projected point
    */
    CV_WRAP Point2f warpPoint(const Point2f &pt, InputArray K, InputArray R) ;

    /** @brief Builds the projection maps according to the given camera data.

    @param src_size Source image size
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param xmap Projection map for the x axis
    @param ymap Projection map for the y axis
    @return Projected image minimum bounding box
    */
    CV_WRAP Rect buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap);

    /** @brief Projects the image.

    @param src Source image
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param interp_mode Interpolation mode
    @param border_mode Border extrapolation mode
    @param dst Projected image
    @return Project image top-left corner
    */
    CV_WRAP Point warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
        CV_OUT OutputArray dst);

    /** @brief Projects the image backward.

    @param src Projected image
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param interp_mode Interpolation mode
    @param border_mode Border extrapolation mode
    @param dst_size Backward-projected image size
    @param dst Backward-projected image
    */
    CV_WRAP void warpBackward(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
        Size dst_size, CV_OUT OutputArray dst);

    /**
    @param src_size Source image bounding box
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @return Projected image minimum bounding box
    */
    CV_WRAP Rect warpRoi(Size src_size, InputArray K, InputArray R);

    CV_WRAP float getScale() const { return 1.f; }
    CV_WRAP void setScale(float) {}
};
}
