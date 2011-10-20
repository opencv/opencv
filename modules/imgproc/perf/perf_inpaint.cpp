#include "perf_precomp.hpp"

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace perf;

CV_ENUM(InpaintingMethod, INPAINT_NS, INPAINT_TELEA)

typedef std::tr1::tuple<Size, InpaintingMethod> InpaintArea_InpaintingMethod_t;
typedef perf::TestBaseWithParam<InpaintArea_InpaintingMethod_t> InpaintArea_InpaintingMethod;


/*
//! restores the damaged image areas using one of the available intpainting algorithms
CV_EXPORTS_W void inpaint( InputArray src, InputArray inpaintMask,
                           OutputArray dst, double inpaintRange, int flags );
*/
PERF_TEST_P( InpaintArea_InpaintingMethod, inpaint,
             testing::Combine(
                 SZ_ALL_SMALL,
                 testing::Values( (int)INPAINT_NS, (int)INPAINT_TELEA ))
           )
{
    Mat src = imread( getDataPath("gpu/hog/road.png") );

    Size sz = std::tr1::get<0>(GetParam());
    int inpaintingMethod = std::tr1::get<1>(GetParam());

    Mat mask(src.size(), CV_8UC1, Scalar(0));

    Rect inpaintArea(src.cols/3, src.rows/3, sz.width, sz.height);
    mask(inpaintArea).setTo(255);

    declare.time(30);

    Mat result;
    TEST_CYCLE(100)
    {
        inpaint(src, mask, result, 10.0, inpaintingMethod);

        //rectangle(result, inpaintArea, Scalar(255));
        //char buf[256];
        //sprintf(buf, "frame_%d_%d.jpg", sz.width, inpaintingMethod);
        //imwrite(buf, result);
    }
}
