#include "precomp.hpp"


cv::Affine3f temp_viz::makeTransformToGlobal(const Vec3f& axis_x, const Vec3f& axis_y, const Vec3f& axis_z, const Vec3f& origin)
{
    Affine3f::Mat3 R;
    R.val[0] = axis_x.val[0];
    R.val[3] = axis_x.val[1];
    R.val[6] = axis_x.val[2];

    R.val[1] = axis_y.val[0];
    R.val[4] = axis_y.val[1];
    R.val[7] = axis_y.val[2];

    R.val[2] = axis_z.val[0];
    R.val[5] = axis_z.val[1];
    R.val[8] = axis_z.val[2];

    return Affine3f(R, origin);
}
