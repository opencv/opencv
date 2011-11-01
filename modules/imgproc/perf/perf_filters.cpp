#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(SobelBorderType, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101)

typedef std::tr1::tuple<Size, std::tr1::tuple<int, int>, int, SobelBorderType> Size_dx_dy_kernel_Border_t;
typedef perf::TestBaseWithParam<Size_dx_dy_kernel_Border_t> Size_dx_dy_kernel_Border;

PERF_TEST_P(Size_dx_dy_kernel_Border, sobel,
            testing::Combine(
                testing::Values(szVGA, szQVGA),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0), make_tuple(1, 1), make_tuple(0, 2), make_tuple(2, 0), make_tuple(2, 2)),
                testing::Values(3, 5),
                testing::ValuesIn(SobelBorderType::all())
            )
          )
{
    Size size = get<0>(GetParam());
    int dx = get<0>(get<1>(GetParam()));
    int dy = get<1>(get<1>(GetParam()));
    int ksize = get<2>(GetParam());
    SobelBorderType border = get<3>(GetParam());

    Mat src(size, CV_8U);
    Mat dst(size, CV_32F);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE(100) { Sobel(src, dst, CV_32F, dx, dy, ksize, 1, 0, border); }

    SANITY_CHECK(dst);
}
