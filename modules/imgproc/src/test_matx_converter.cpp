#include "precomp.hpp"

namespace cv{
    void testMatxPythonConverter(InputArray _src, OutputArray _dst, const Vec2d& defaultParam){
        printf("%f %f\n", defaultParam[0], defaultParam[1]);
        Mat src = _src.getMat();
        src.copyTo(_dst);
    }
}
