#include <opencv2/viz/types.hpp>



//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::Color

temp_viz::Color::Color() : Scalar(0, 0, 0) {}
temp_viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
temp_viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
temp_viz::Color::Color(const Scalar& color) : Scalar(color) {}

temp_viz::Color temp_viz::Color::black()   { return Color(  0,   0, 0); }
temp_viz::Color temp_viz::Color::green()   { return Color(  0, 255, 0); }
temp_viz::Color temp_viz::Color::blue()    { return Color(255,   0, 0); }
temp_viz::Color temp_viz::Color::cyan()    { return Color(255, 255, 0); }

temp_viz::Color temp_viz::Color::red()     { return Color(  0,   0, 255); }
temp_viz::Color temp_viz::Color::magenta() { return Color(  0, 255, 255); }
temp_viz::Color temp_viz::Color::yellow()  { return Color(255,   0, 255); }
temp_viz::Color temp_viz::Color::white()   { return Color(255, 255, 255); }

temp_viz::Color temp_viz::Color::gray()    { return Color(128, 128, 128); }

temp_viz::Point3d temp_viz::operator*(const temp_viz::Affine3f& affine, const temp_viz::Point3d& point)
{
    const temp_viz::Matx44f& m = affine.matrix;
    temp_viz::Point3d result;
    result.x = m.val[0] * point.x + m.val[1] * point.y + m.val[ 2] * point.z + m.val[ 3];
    result.y = m.val[4] * point.x + m.val[5] * point.y + m.val[ 6] * point.z + m.val[ 7];
    result.z = m.val[8] * point.x + m.val[9] * point.y + m.val[10] * point.z + m.val[11];
    return result;
}