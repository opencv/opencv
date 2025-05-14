// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

    static const int fixedShiftU8 = 8;

    template <typename T, int fixedShift>
    void eval4(int64_t   xcoeff0, int64_t   xcoeff1, int64_t   ycoeff0, int64_t   ycoeff1, int cn,
               uint8_t* src_pt00, uint8_t* src_pt01, uint8_t* src_pt10, uint8_t* src_pt11, uint8_t* dst_pt)
    {
        static const int64_t fixedRound = ((1LL << (fixedShift * 2)) >> 1);
        int64_t val = (((T*)src_pt00)[cn] * xcoeff0 + ((T*)src_pt01)[cn] * xcoeff1) * ycoeff0 +
                      (((T*)src_pt10)[cn] * xcoeff0 + ((T*)src_pt11)[cn] * xcoeff1) * ycoeff1 ;
        ((T*)dst_pt)[cn] = saturate_cast<T>((val + fixedRound) >> (fixedShift * 2));
    }

TEST(Resize_Bitexact, Linear8U)
{
    static const int64_t fixedOne = (1L << fixedShiftU8);

    struct testmode
    {
        int type;
        Size sz;
    } modes[] = {
        { CV_8UC1, Size( 512, 768) }, //   1/2       1
        { CV_8UC3, Size( 512, 768) },
        { CV_8UC1, Size(1024, 384) }, //    1       1/2
        { CV_8UC4, Size(1024, 384) },
        { CV_8UC1, Size( 512, 384) }, //   1/2      1/2
        { CV_8UC2, Size( 512, 384) },
        { CV_8UC3, Size( 512, 384) },
        { CV_8UC4, Size( 512, 384) },
        { CV_8UC1, Size( 256, 192) }, //   1/4      1/4
        { CV_8UC2, Size( 256, 192) },
        { CV_8UC3, Size( 256, 192) },
        { CV_8UC4, Size( 256, 192) },
        { CV_8UC1, Size(   4,   3) }, //   1/256    1/256
        { CV_8UC2, Size(   4,   3) },
        { CV_8UC3, Size(   4,   3) },
        { CV_8UC4, Size(   4,   3) },
        { CV_8UC1, Size( 342, 384) }, //   1/3      1/2
        { CV_8UC1, Size( 342, 256) }, //   1/3      1/3
        { CV_8UC2, Size( 342, 256) },
        { CV_8UC3, Size( 342, 256) },
        { CV_8UC4, Size( 342, 256) },
        { CV_8UC1, Size( 512, 256) }, //   1/2      1/3
        { CV_8UC1, Size( 146, 110) }, //   1/7      1/7
        { CV_8UC3, Size( 146, 110) },
        { CV_8UC4, Size( 146, 110) },
        { CV_8UC1, Size( 931, 698) }, //  10/11    10/11
        { CV_8UC2, Size( 931, 698) },
        { CV_8UC3, Size( 931, 698) },
        { CV_8UC4, Size( 931, 698) },
        { CV_8UC1, Size( 853, 640) }, //  10/12    10/12
        { CV_8UC3, Size( 853, 640) },
        { CV_8UC4, Size( 853, 640) },
        { CV_8UC1, Size(1004, 753) }, // 251/256  251/256
        { CV_8UC2, Size(1004, 753) },
        { CV_8UC3, Size(1004, 753) },
        { CV_8UC4, Size(1004, 753) },
        { CV_8UC1, Size(2048,1536) }, //    2        2
        { CV_8UC2, Size(2048,1536) },
        { CV_8UC4, Size(2048,1536) },
        { CV_8UC1, Size(3072,2304) }, //    3        3
        { CV_8UC3, Size(3072,2304) },
        { CV_8UC1, Size(7168,5376) }  //    7        7
    };

    for (int modeind = 0, _modecnt = sizeof(modes) / sizeof(modes[0]); modeind < _modecnt; ++modeind)
        {
            int type = modes[modeind].type, depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
            int dcols = modes[modeind].sz.width, drows = modes[modeind].sz.height;
            int cols = 1024, rows = 768;

            double inv_scale_x = (double)dcols / cols;
            double inv_scale_y = (double)drows / rows;
            softdouble scale_x = softdouble::one() / softdouble(inv_scale_x);
            softdouble scale_y = softdouble::one() / softdouble(inv_scale_y);

            Mat src(rows, cols, type), refdst(drows, dcols, type), dst;
            RNG rnd(0x123456789abcdefULL);
            for (int j = 0; j < rows; j++)
            {
                uint8_t* line = src.ptr(j);
                for (int i = 0; i < cols; i++)
                    for (int c = 0; c < cn; c++)
                    {
                        double val = j < rows / 2 ? ( i < cols / 2 ? ((sin((i + 1)*CV_PI / 256.)*sin((j + 1)*CV_PI / 256.)*sin((cn + 4)*CV_PI / 8.) + 1.)*128.)                         :
                                                                     (((i / 128 + j / 128) % 2) * 250 + (j / 128) % 2)                                                                ) :
                                                    ( i < cols / 2 ? ((i / 128) * (85 - j / 256 * 40) * ((j / 128) % 2) + (7 - i / 128) * (85 - j / 256 * 40) * ((j / 128 + 1) % 2))    :
                                                                     ((uchar)rnd)                                                                                                     ) ;
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

            for (int j = 0; j < drows; j++)
            {
                softdouble src_row_flt = scale_y*(softdouble(j) + softdouble(0.5)) - softdouble(0.5);
                int src_row = cvFloor(src_row_flt);
                int64_t ycoeff1 = cvRound64((src_row_flt - softdouble(src_row))*softdouble(fixedOne));
                int64_t ycoeff0 = fixedOne - ycoeff1;

                for (int i = 0; i < dcols; i++)
                {
                    softdouble src_col_flt = scale_x*(softdouble(i) + softdouble(0.5)) - softdouble(0.5);
                    int src_col = cvFloor(src_col_flt);
                    int64_t xcoeff1 = cvRound64((src_col_flt - softdouble(src_col))*softdouble(fixedOne));
                    int64_t xcoeff0 = fixedOne - xcoeff1;

                    uint8_t* dst_pt = refdst.ptr(j, i);
                    uint8_t* src_pt00 = src.ptr( src_row      < 0 ? 0 :  src_row      >= rows ? rows - 1 :  src_row     ,
                                                 src_col      < 0 ? 0 :  src_col      >= cols ? cols - 1 :  src_col     );
                    uint8_t* src_pt01 = src.ptr( src_row      < 0 ? 0 :  src_row      >= rows ? rows - 1 :  src_row     ,
                                                (src_col + 1) < 0 ? 0 : (src_col + 1) >= cols ? cols - 1 : (src_col + 1));
                    uint8_t* src_pt10 = src.ptr((src_row + 1) < 0 ? 0 : (src_row + 1) >= rows ? rows - 1 : (src_row + 1),
                                                 src_col      < 0 ? 0 :  src_col      >= cols ? cols - 1 :  src_col     );
                    uint8_t* src_pt11 = src.ptr((src_row + 1) < 0 ? 0 : (src_row + 1) >= rows ? rows - 1 : (src_row + 1),
                                                (src_col + 1) < 0 ? 0 : (src_col + 1) >= cols ? cols - 1 : (src_col + 1));
                    for (int c = 0; c < cn; c++)
                    {
                        if (depth == CV_8U)
                            eval4< uint8_t, fixedShiftU8>(xcoeff0, xcoeff1, ycoeff0, ycoeff1, c, src_pt00, src_pt01, src_pt10, src_pt11, dst_pt);
                        else if (depth == CV_16U)
                            eval4<uint16_t, fixedShiftU8>(xcoeff0, xcoeff1, ycoeff0, ycoeff1, c, src_pt00, src_pt01, src_pt10, src_pt11, dst_pt);
                        else if (depth == CV_16S)
                            eval4< int16_t, fixedShiftU8>(xcoeff0, xcoeff1, ycoeff0, ycoeff1, c, src_pt00, src_pt01, src_pt10, src_pt11, dst_pt);
                        else if (depth == CV_32S)
                            eval4< int32_t, fixedShiftU8>(xcoeff0, xcoeff1, ycoeff0, ycoeff1, c, src_pt00, src_pt01, src_pt10, src_pt11, dst_pt);
                        else
                            CV_Assert(0);
                    }
                }
            }

            cv::resize(src, dst, Size(dcols, drows), 0, 0, cv::INTER_LINEAR_EXACT);
            EXPECT_GE(0, cvtest::norm(refdst, dst, cv::NORM_L1))
                << "Resize " << cn << "-chan mat from " << cols << "x" << rows << " to " << dcols << "x" << drows << " failed with max diff " << cvtest::norm(refdst, dst, cv::NORM_INF);
        }
}

PARAM_TEST_CASE(Resize_Bitexact, int)
{
public:
    int depth;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
    }

    double CountDiff(const Mat& src)
    {
        Mat dstExact; cv::resize(src, dstExact, Size(), 2, 1, INTER_NEAREST_EXACT);
        Mat dstNonExact; cv::resize(src, dstNonExact, Size(), 2, 1, INTER_NEAREST);

        return cv::norm(dstExact, dstNonExact, NORM_INF);
    }
};

TEST_P(Resize_Bitexact, Nearest8U_vsNonExact)
{
    Mat mat_color, mat_gray;
    Mat src_color = imread(cvtest::findDataFile("shared/lena.png"));
    Mat src_gray; cv::cvtColor(src_color, src_gray, COLOR_BGR2GRAY);
    src_color.convertTo(mat_color, depth);
    src_gray.convertTo(mat_gray, depth);

    EXPECT_EQ(CountDiff(mat_color), 0) << "color, type: " << depth;
    EXPECT_EQ(CountDiff(mat_gray), 0) << "gray, type: " << depth;
}

// Now INTER_NEAREST's convention and INTER_NEAREST_EXACT's one are different.
INSTANTIATE_TEST_CASE_P(DISABLED_Imgproc, Resize_Bitexact,
    testing::Values(CV_8U, CV_16U, CV_32F, CV_64F)
);

TEST(Resize_Bitexact, Nearest8U)
{
    Mat src[6], dst[6];

    // 2x decimation
    src[0] = (Mat_<uint8_t>(1, 6) << 0, 1, 2, 3, 4, 5);
    dst[0] = (Mat_<uint8_t>(1, 3) << 1, 3, 5);

    // decimation odd to 1
    src[1] = (Mat_<uint8_t>(1, 5) << 0, 1, 2, 3, 4);
    dst[1] = (Mat_<uint8_t>(1, 1) << 2);

    // decimation n*2-1 to n
    src[2] = (Mat_<uint8_t>(1, 5) << 0, 1, 2, 3, 4);
    dst[2] = (Mat_<uint8_t>(1, 3) << 0, 2, 4);

    // decimation n*2+1 to n
    src[3] = (Mat_<uint8_t>(1, 5) << 0, 1, 2, 3, 4);
    dst[3] = (Mat_<uint8_t>(1, 2) << 1, 3);

    // zoom
    src[4] = (Mat_<uint8_t>(3, 5) <<
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14);
    dst[4] = (Mat_<uint8_t>(5, 7) <<
        0, 1, 1, 2, 3, 3, 4,
        0, 1, 1, 2, 3, 3, 4,
        5, 6, 6, 7, 8, 8, 9,
        10, 11, 11, 12, 13, 13, 14,
        10, 11, 11, 12, 13, 13, 14);

    src[5] = (Mat_<uint8_t>(2, 3) <<
        0, 1, 2,
        3, 4, 5);
    dst[5] = (Mat_<uint8_t>(4, 6) <<
        0, 0, 1, 1, 2, 2,
        0, 0, 1, 1, 2, 2,
        3, 3, 4, 4, 5, 5,
        3, 3, 4, 4, 5, 5);

    for (int i = 0; i < 6; i++)
    {
        Mat calc;
        resize(src[i], calc, dst[i].size(), 0, 0, INTER_NEAREST_EXACT);
        EXPECT_EQ(cvtest::norm(calc, dst[i], cv::NORM_L1), 0);

        resize(src[i].t(), calc, dst[i].t().size(), 0, 0, INTER_NEAREST_EXACT);
        EXPECT_EQ(cvtest::norm(calc, dst[i].t(), cv::NORM_L1), 0);
    }
}

}} // namespace
