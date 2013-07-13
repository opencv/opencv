#include <opencv2/viz/types.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

cv::viz::Color::Color() : Scalar(0, 0, 0) {}
cv::viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
cv::viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0, 0); }
cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255, 0); }
cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0, 0); }
cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255, 0); }

cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
cv::viz::Color cv::viz::Color::magenta() { return Color(  0, 255, 255); }
cv::viz::Color cv::viz::Color::yellow()  { return Color(255,   0, 255); }
cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }

cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }

