#include "perf_precomp.hpp"

namespace opencv_test
{

CV_ENUM(InpaintingMethod, INPAINT_NS, INPAINT_TELEA)
typedef tuple<Size, InpaintingMethod> InpaintArea_InpaintingMethod_t;
typedef perf::TestBaseWithParam<InpaintArea_InpaintingMethod_t> InpaintArea_InpaintingMethod;


PERF_TEST_P(InpaintArea_InpaintingMethod, inpaint,
            testing::Combine(
                testing::Values(::perf::szSmall24, ::perf::szSmall32, ::perf::szSmall64),
                InpaintingMethod::all()
                )
            )
{
    Mat src = imread(getDataPath("gpu/hog/road.png"));

    Size sz = get<0>(GetParam());
    int inpaintingMethod = get<1>(GetParam());

    Mat mask(src.size(), CV_8UC1, Scalar(0));
    Mat result(src.size(), src.type());

    Rect inpaintArea(src.cols/3, src.rows/3, sz.width, sz.height);
    mask(inpaintArea).setTo(255);

    declare.in(src, mask).out(result).time(120);

    TEST_CYCLE() inpaint(src, mask, result, 10.0, inpaintingMethod);

    Mat inpaintedArea = result(inpaintArea);
    SANITY_CHECK(inpaintedArea);
}

} // namespace
