// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace
{
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
}

TEST(Resize_Bitexact, Linear8U)
{
    static const int64_t fixedOne = (1L << fixedShiftU8);

    int types[] = { CV_8UC1, CV_8UC4 };
    // NOTICE: 2x downscaling ommitted since it use different rounding
    //                      1/2   1           1   1/2        1/2  1/2        1/4  1/4   1/256 1/256      1/3  1/2        1/3  1/3        1/2  1/3        1/7  1/7
    Size dstsizes[] = {Size(512, 768), Size(1024, 384), Size(512, 384), Size(256, 192), Size(4, 3), Size(342, 384), Size(342, 256), Size(512, 256), Size(146, 110),
    //                    10/11 10/11     10/12 10/12         251/256          2    2           3    3           7    7
                       Size(931, 698), Size(853, 640), Size(1004, 753), Size(2048,1536), Size(3072,2304), Size(7168,5376) };

    for (int dsizeind = 0, _dsizecnt = sizeof(dstsizes) / sizeof(dstsizes[0]); dsizeind < _dsizecnt; ++dsizeind)
        for (int typeind = 0, _typecnt = sizeof(types) / sizeof(types[0]); typeind < _typecnt; ++typeind)
        {
            int type = types[typeind], depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
            int dcols = dstsizes[dsizeind].width, drows = dstsizes[dsizeind].height;
            int cols = 1024, rows = 768;

            double inv_scale_x = (double)dcols / cols;
            double inv_scale_y = (double)drows / rows;
            softdouble scale_x = softdouble::one() / softdouble(inv_scale_x);
            softdouble scale_y = softdouble::one() / softdouble(inv_scale_y);

            Mat src(rows, cols, type), refdst(drows, dcols, type), dst;
            for (int j = 0; j < rows; j++)
            {
                uint8_t* line = src.ptr(j);
                for (int i = 0; i < cols; i++)
                    for (int c = 0; c < cn; c++)
                    {
                        RNG rnd(0x123456789abcdefULL);
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
                << "Resize from " << cols << "x" << rows << " to " << dcols << "x" << drows << " failed with max diff " << cvtest::norm(refdst, dst, cv::NORM_INF);
        }
}

///* End of file. */
