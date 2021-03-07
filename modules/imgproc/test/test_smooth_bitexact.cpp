// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

    static const int fixedShiftU8 = 8;
    static const int64_t fixedOneU8 = (1L << fixedShiftU8);
    static const int fixedShiftU16 = 16;
    static const int64_t fixedOneU16 = (1L << fixedShiftU16);

    int64_t vU8[][9] = {
        { fixedOneU8 }, // size 1, sigma 0
        { fixedOneU8 >> 2, fixedOneU8 >> 1, fixedOneU8 >> 2 }, // size 3, sigma 0
        { fixedOneU8 >> 4, fixedOneU8 >> 2, 6 * (fixedOneU8 >> 4), fixedOneU8 >> 2, fixedOneU8 >> 4 }, // size 5, sigma 0
        { fixedOneU8 >> 5, 7 * (fixedOneU8 >> 6), 7 * (fixedOneU8 >> 5), 9 * (fixedOneU8 >> 5), 7 * (fixedOneU8 >> 5), 7 * (fixedOneU8 >> 6), fixedOneU8 >> 5 }, // size 7, sigma 0
        { 4, 13, 30, 51, 60, 51, 30, 13, 4 }, // size 9, sigma 0
#if 1
#define CV_TEST_INACCURATE_GAUSSIAN_BLUR
        { 81, 94, 81 }, // size 3, sigma 1.75
        { 65, 126, 65 }, // size 3, sigma 0.875
        { 0, 7, 242, 7, 0 }, // size 5, sigma 0.375
        { 4, 56, 136, 56, 4 } // size 5, sigma 0.75
#endif
    };

    int64_t vU16[][9] = {
        { fixedOneU16 }, // size 1, sigma 0
        { fixedOneU16 >> 2, fixedOneU16 >> 1, fixedOneU16 >> 2 }, // size 3, sigma 0
        { fixedOneU16 >> 4, fixedOneU16 >> 2, 6 * (fixedOneU16 >> 4), fixedOneU16 >> 2, fixedOneU16 >> 4 }, // size 5, sigma 0
        { fixedOneU16 >> 5, 7 * (fixedOneU16 >> 6), 7 * (fixedOneU16 >> 5), 9 * (fixedOneU16 >> 5), 7 * (fixedOneU16 >> 5), 7 * (fixedOneU16 >> 6), fixedOneU16 >> 5 }, // size 7, sigma 0
        { 4<<8, 13<<8, 30<<8, 51<<8, 60<<8, 51<<8, 30<<8, 13<<8, 4<<8 } // size 9, sigma 0
    };

    template <typename T, int fixedShift>
    T eval(Mat src, vector<int64_t> kernelx, vector<int64_t> kernely)
    {
        static const int64_t fixedRound = ((1LL << (fixedShift * 2)) >> 1);
        int64_t val = 0;
        for (size_t j = 0; j < kernely.size(); j++)
        {
            int64_t lineval = 0;
            for (size_t i = 0; i < kernelx.size(); i++)
                lineval += src.at<T>((int)j, (int)i) * kernelx[i];
            val += lineval * kernely[j];
        }
        return saturate_cast<T>((val + fixedRound) >> (fixedShift * 2));
    }

    struct testmode
    {
        int type;
        Size sz;
        Size kernel;
        double sigma_x;
        double sigma_y;
        vector<int64_t> kernel_x;
        vector<int64_t> kernel_y;
    };

    int bordermodes[] = {
        BORDER_CONSTANT | BORDER_ISOLATED,
        BORDER_REPLICATE | BORDER_ISOLATED,
        BORDER_REFLECT | BORDER_ISOLATED,
        BORDER_WRAP | BORDER_ISOLATED,
        BORDER_REFLECT_101 | BORDER_ISOLATED
//        BORDER_CONSTANT,
//        BORDER_REPLICATE,
//        BORDER_REFLECT,
//        BORDER_WRAP,
//        BORDER_REFLECT_101
    };

    template <int fixedShift>
    void checkMode(const testmode& mode)
    {
        int type = mode.type, depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
        int dcols = mode.sz.width, drows = mode.sz.height;
        Size kernel = mode.kernel;

        int rows = drows + 20, cols = dcols + 20;
        Mat src(rows, cols, type), refdst(drows, dcols, type), dst;
        for (int j = 0; j < rows; j++)
        {
            uint8_t* line = src.ptr(j);
            for (int i = 0; i < cols; i++)
                for (int c = 0; c < cn; c++)
                {
                    RNG rnd(0x123456789abcdefULL);
                    double val = j < rows / 2 ? (i < cols / 2 ? ((sin((i + 1)*CV_PI / 256.)*sin((j + 1)*CV_PI / 256.)*sin((cn + 4)*CV_PI / 8.) + 1.)*128.) :
                        (((i / 128 + j / 128) % 2) * 250 + (j / 128) % 2)) :
                        (i < cols / 2 ? ((i / 128) * (85 - j / 256 * 40) * ((j / 128) % 2) + (7 - i / 128) * (85 - j / 256 * 40) * ((j / 128 + 1) % 2)) :
                        ((uchar)rnd));
                    if (depth == CV_8U)
                        line[i*cn + c] = (uint8_t)val;
                    else if (depth == CV_16U)
                        ((uint16_t*)line)[i*cn + c] = (uint16_t)val;
                    else if (depth == CV_16S)
                        ((int16_t*)line)[i*cn + c] = (int16_t)val;
                    else if (depth == CV_32S)
                        ((int32_t*)line)[i*cn + c] = (int32_t)val;
                    else
                        CV_Assert(0);
                }
        }
        Mat src_roi = src(Rect(10, 10, dcols, drows));


        for (int borderind = 0, _bordercnt = sizeof(bordermodes) / sizeof(bordermodes[0]); borderind < _bordercnt; ++borderind)
        {
            Mat src_border;
            cv::copyMakeBorder(src_roi, src_border, kernel.height / 2, kernel.height / 2, kernel.width / 2, kernel.width / 2, bordermodes[borderind]);
            for (int c = 0; c < src_border.channels(); c++)
            {
                int fromTo[2] = { c, 0 };
                int toFrom[2] = { 0, c };
                Mat src_chan(src_border.size(), CV_MAKETYPE(src_border.depth(),1));
                Mat dst_chan(refdst.size(), CV_MAKETYPE(refdst.depth(), 1));
                mixChannels(src_border, src_chan, fromTo, 1);
                for (int j = 0; j < drows; j++)
                    for (int i = 0; i < dcols; i++)
                    {
                        if (depth == CV_8U)
                            dst_chan.at<uint8_t>(j, i) = eval<uint8_t, fixedShift>(src_chan(Rect(i,j,kernel.width,kernel.height)), mode.kernel_x, mode.kernel_y);
                        else if (depth == CV_16U)
                            dst_chan.at<uint16_t>(j, i) = eval<uint16_t, fixedShift>(src_chan(Rect(i, j, kernel.width, kernel.height)), mode.kernel_x, mode.kernel_y);
                        else if (depth == CV_16S)
                            dst_chan.at<int16_t>(j, i) = eval<int16_t, fixedShift>(src_chan(Rect(i, j, kernel.width, kernel.height)), mode.kernel_x, mode.kernel_y);
                        else if (depth == CV_32S)
                            dst_chan.at<int32_t>(j, i) = eval<int32_t, fixedShift>(src_chan(Rect(i, j, kernel.width, kernel.height)), mode.kernel_x, mode.kernel_y);
                        else
                            CV_Assert(0);
                    }
                mixChannels(dst_chan, refdst, toFrom, 1);
            }

            cv::GaussianBlur(src_roi, dst, kernel, mode.sigma_x, mode.sigma_y, bordermodes[borderind]);

            EXPECT_GE(0, cvtest::norm(refdst, dst, cv::NORM_L1))
                << "GaussianBlur " << cn << "-chan mat " << drows << "x" << dcols << " by kernel " << kernel << " sigma(" << mode.sigma_x << ";" << mode.sigma_y << ") failed with max diff " << cvtest::norm(refdst, dst, cv::NORM_INF);
        }
    }

TEST(GaussianBlur_Bitexact, Linear8U)
{
    testmode modes[] = {
        { CV_8UC1, Size(   1,   1), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   2,   2), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   3,   1), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   1,   3), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   3,   3), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   3,   3), Size(5, 5), 0, 0, vector<int64_t>(vU8[2], vU8[2]+5), vector<int64_t>(vU8[2], vU8[2]+5) },
        { CV_8UC1, Size(   3,   3), Size(7, 7), 0, 0, vector<int64_t>(vU8[3], vU8[3]+7), vector<int64_t>(vU8[3], vU8[3]+7) },
        { CV_8UC1, Size(   5,   5), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size(   5,   5), Size(5, 5), 0, 0, vector<int64_t>(vU8[2], vU8[2]+5), vector<int64_t>(vU8[2], vU8[2]+5) },
        { CV_8UC1, Size(   3,   5), Size(5, 5), 0, 0, vector<int64_t>(vU8[2], vU8[2]+5), vector<int64_t>(vU8[2], vU8[2]+5) },
        { CV_8UC1, Size(   5,   5), Size(5, 5), 0, 0, vector<int64_t>(vU8[2], vU8[2]+5), vector<int64_t>(vU8[2], vU8[2]+5) },
        { CV_8UC1, Size(   5,   5), Size(7, 7), 0, 0, vector<int64_t>(vU8[3], vU8[3]+7), vector<int64_t>(vU8[3], vU8[3]+7) },
        { CV_8UC1, Size(   7,   7), Size(7, 7), 0, 0, vector<int64_t>(vU8[3], vU8[3]+7), vector<int64_t>(vU8[3], vU8[3]+7) },
        { CV_8UC1, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC2, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC3, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC4, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU8[1], vU8[1]+3), vector<int64_t>(vU8[1], vU8[1]+3) },
        { CV_8UC1, Size( 256, 128), Size(5, 5), 0, 0, vector<int64_t>(vU8[2], vU8[2]+5), vector<int64_t>(vU8[2], vU8[2]+5) },
        { CV_8UC1, Size( 256, 128), Size(7, 7), 0, 0, vector<int64_t>(vU8[3], vU8[3]+7), vector<int64_t>(vU8[3], vU8[3]+7) },
        { CV_8UC1, Size( 256, 128), Size(9, 9), 0, 0, vector<int64_t>(vU8[4], vU8[4]+9), vector<int64_t>(vU8[4], vU8[4]+9) },
#ifdef CV_TEST_INACCURATE_GAUSSIAN_BLUR
        { CV_8UC1, Size( 256, 128), Size(3, 3), 1.75, 0.875, vector<int64_t>(vU8[5], vU8[5]+3), vector<int64_t>(vU8[6], vU8[6]+3) },
        { CV_8UC2, Size( 256, 128), Size(3, 3), 1.75, 0.875, vector<int64_t>(vU8[5], vU8[5]+3), vector<int64_t>(vU8[6], vU8[6]+3) },
        { CV_8UC3, Size( 256, 128), Size(3, 3), 1.75, 0.875, vector<int64_t>(vU8[5], vU8[5]+3), vector<int64_t>(vU8[6], vU8[6]+3) },
        { CV_8UC4, Size( 256, 128), Size(3, 3), 1.75, 0.875, vector<int64_t>(vU8[5], vU8[5]+3), vector<int64_t>(vU8[6], vU8[6]+3) },
        { CV_8UC1, Size( 256, 128), Size(5, 5), 0.375, 0.75, vector<int64_t>(vU8[7], vU8[7]+5), vector<int64_t>(vU8[8], vU8[8]+5) }
#endif
    };

    for (int modeind = 0, _modecnt = sizeof(modes) / sizeof(modes[0]); modeind < _modecnt; ++modeind)
    {
        checkMode<fixedShiftU8>(modes[modeind]);
    }
}

TEST(GaussianBlur_Bitexact, Linear16U)
{
        testmode modes[] = {
        { CV_16UC1, Size(   1,   1), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   2,   2), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   3,   1), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   1,   3), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   3,   3), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   3,   3), Size(5, 5), 0, 0, vector<int64_t>(vU16[2], vU16[2]+5), vector<int64_t>(vU16[2], vU16[2]+5) },
        { CV_16UC1, Size(   3,   3), Size(7, 7), 0, 0, vector<int64_t>(vU16[3], vU16[3]+7), vector<int64_t>(vU16[3], vU16[3]+7) },
        { CV_16UC1, Size(   5,   5), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size(   5,   5), Size(5, 5), 0, 0, vector<int64_t>(vU16[2], vU16[2]+5), vector<int64_t>(vU16[2], vU16[2]+5) },
        { CV_16UC1, Size(   3,   5), Size(5, 5), 0, 0, vector<int64_t>(vU16[2], vU16[2]+5), vector<int64_t>(vU16[2], vU16[2]+5) },
        { CV_16UC1, Size(   5,   5), Size(5, 5), 0, 0, vector<int64_t>(vU16[2], vU16[2]+5), vector<int64_t>(vU16[2], vU16[2]+5) },
        { CV_16UC1, Size(   5,   5), Size(7, 7), 0, 0, vector<int64_t>(vU16[3], vU16[3]+7), vector<int64_t>(vU16[3], vU16[3]+7) },
        { CV_16UC1, Size(   7,   7), Size(7, 7), 0, 0, vector<int64_t>(vU16[3], vU16[3]+7), vector<int64_t>(vU16[3], vU16[3]+7) },
        { CV_16UC1, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC2, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC3, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC4, Size( 256, 128), Size(3, 3), 0, 0, vector<int64_t>(vU16[1], vU16[1]+3), vector<int64_t>(vU16[1], vU16[1]+3) },
        { CV_16UC1, Size( 256, 128), Size(5, 5), 0, 0, vector<int64_t>(vU16[2], vU16[2]+5), vector<int64_t>(vU16[2], vU16[2]+5) },
        { CV_16UC1, Size( 256, 128), Size(7, 7), 0, 0, vector<int64_t>(vU16[3], vU16[3]+7), vector<int64_t>(vU16[3], vU16[3]+7) },
        { CV_16UC1, Size( 256, 128), Size(9, 9), 0, 0, vector<int64_t>(vU16[4], vU16[4]+9), vector<int64_t>(vU16[4], vU16[4]+9) },
    };

    for (int modeind = 0, _modecnt = sizeof(modes) / sizeof(modes[0]); modeind < _modecnt; ++modeind)
    {
        checkMode<16>(modes[modeind]);
    }
}

TEST(GaussianBlur_Bitexact, regression_15015)
{
    Mat src(100,100,CV_8UC3,Scalar(255,255,255));
    Mat dst;
    GaussianBlur(src, dst, Size(5, 5), 0);
    ASSERT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));
}


static void checkGaussianBlur_8Uvs32F(const Mat& src8u, const Mat& src32f, int N, double sigma)
{
    Mat dst8u; GaussianBlur(src8u, dst8u, Size(N, N), sigma);     // through bit-exact path
    Mat dst8u_32f; dst8u.convertTo(dst8u_32f, CV_32F);

    Mat dst32f; GaussianBlur(src32f, dst32f, Size(N, N), sigma);  // without bit-exact computations

    double normINF_32f = cv::norm(dst8u_32f, dst32f, NORM_INF);
    EXPECT_LE(normINF_32f, 1.0);
}

TEST(GaussianBlur_Bitexact, regression_9863)
{
    Mat src8u = imread(cvtest::findDataFile("shared/lena.png"));
     Mat src32f; src8u.convertTo(src32f, CV_32F);

    checkGaussianBlur_8Uvs32F(src8u, src32f, 151, 30);
}

}} // namespace
