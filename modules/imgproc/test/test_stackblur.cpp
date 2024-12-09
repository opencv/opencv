// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
StackBlur - a fast almost Gaussian Blur
Theory: http://underdestruction.com/2004/02/25/stackblur-2004
The code has been borrowed from (https://github.com/flozz/StackBlur).

Below is the original copyright
*/

/*
Copyright (c) 2010 Mario Klingemann

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
 */


#include "test_precomp.hpp"

namespace opencv_test { namespace {

template<typename T>
void _stackblurRef(const Mat& src, Mat& dst, Size ksize)
{
    CV_Assert(!src.empty());
    CV_Assert(ksize.width > 0 && ksize.height > 0 && ksize.height % 2 == 1 && ksize.width % 2 == 1);

    dst.create(src.size(), src.type());
    const int CN = src.channels();

    int rowsImg = src.rows;
    int colsImg = src.cols;
    int wm = colsImg - 1;

    int radiusW = ksize.width / 2;
    int stackLenW = ksize.width;
    const float mulW = 1.0f / (((float )radiusW + 1.0f) * ((float )radiusW + 1.0f));

    // Horizontal direction
    std::vector<T> stack(stackLenW * CN);
    for (int row = 0; row < rowsImg; row++)
    {
        std::vector<float> sum(CN, 0);
        std::vector<float> sumIn(CN, 0);
        std::vector<float> sumOut(CN, 0);

        const T* srcPtr = src.ptr<T>(row);

        for (int i = 0; i <= radiusW; i++)
        {
            for (int ci = 0; ci < CN; ci++)
            {
                T tmp = *(srcPtr + ci);
                stack[i * CN + ci] = tmp;
                sum[ci] += tmp * (i + 1);
                sumOut[ci] += tmp;
            }
        }

        for (int i = 1; i <= radiusW; i++)
        {
            if (i <= wm) srcPtr += CN;
            for(int ci = 0; ci < CN; ci++)
            {
                T tmp = *(srcPtr + ci);
                stack[(i + radiusW) * CN + ci] = tmp;
                sum[ci] += tmp * (radiusW + 1 - i);
                sumIn[ci] += tmp;
            }
        }

        int sp = radiusW;
        int xp = radiusW ;
        if (xp > wm) xp = wm;

        T* dstPtr = dst.ptr<T>(row);
        srcPtr = src.ptr<T>(row) + xp * CN;

        int stackStart= 0;

        for (int i = 0; i < colsImg; i++)
        {
            stackStart = sp + stackLenW - radiusW;

            if (stackStart >= stackLenW) stackStart -= stackLenW;

            for(int ci = 0; ci < CN; ci++)
            {
                *(dstPtr + ci) = cv::saturate_cast<T>(sum[ci] * mulW);
                sum[ci] -= sumOut[ci];
                sumOut[ci] -= stack[stackStart*CN + ci];
            }

            const T* srcNew = srcPtr;

            if(xp < wm)
                srcNew += CN;

            for (int ci = 0; ci < CN; ci++)
            {
                stack[stackStart * CN + ci] = *(srcNew + ci);
                sumIn[ci] += *(srcNew + ci);
                sum[ci] += sumIn[ci];
            }

            int sp1 = sp + 1;
            if (sp1 >= stackLenW)
                sp1 = 0;

            for(int ci = 0; ci < CN; ci++)
            {
                T tmp = stack[sp1*CN + ci];
                sumOut[ci] += tmp;
                sumIn[ci] -= tmp;
            }

            dstPtr += CN;

            if (xp < wm)
            {
                xp++;
                srcPtr += CN;
            }

            ++sp;
            if (sp >= stackLenW)
                sp = 0;
        }
    }

    // Vertical direction
    int hm = rowsImg - 1;
    int widthElem = colsImg * CN;
    int radiusH = ksize.height / 2;
    int stackLenH = ksize.height;
    const float mulH = 1.0f / (((float )radiusH + 1.0f) * ((float )radiusH + 1.0f));

    stack.resize(stackLenH, 0);
    for (int col = 0; col < widthElem; col++)
    {
        const T* srcPtr =dst.ptr<T>() + col;
        float sum0 = 0;
        float sumIn0 = 0;
        float sumOut0 = 0;

        for (int i = 0; i <= radiusH; i++)
        {
            T tmp = (T)(*srcPtr);
            stack[i] = tmp;
            sum0 += tmp * (i + 1);
            sumOut0 += tmp;
        }

        for (int i = 1; i <= radiusH; i++)
        {
            if (i <= hm) srcPtr += widthElem;
            T tmp = (T)(*srcPtr);
            stack[i + radiusH] = tmp;
            sum0 += tmp * (radiusH - i + 1);
            sumIn0 += tmp;
        }

        int sp = radiusH;
        int yp = radiusH;

        if (yp > hm) yp = hm;

        T* dstPtr = dst.ptr<T>() + col;
        srcPtr = dst.ptr<T>(yp) + col;

        const T* srcNew;

        int stackStart = 0;

        for (int i = 0; i < rowsImg; i++)
        {
            stackStart = sp + stackLenH - radiusH;
            if (stackStart >= stackLenH) stackStart -= stackLenH;

            *(dstPtr) = saturate_cast<T>(sum0 * mulH);
            sum0 -= sumOut0;
            sumOut0 -= stack[stackStart];
            srcNew = srcPtr;

            if (yp < hm)
                srcNew += widthElem;

            stack[stackStart] = *(srcNew);
            sumIn0 += *(srcNew);
            sum0 += sumIn0;

            int sp1 = sp + 1;
            sp1 &= -(sp1 < stackLenH);

            sumOut0 += stack[sp1];
            sumIn0 -= stack[sp1];

            dstPtr += widthElem;

            if (yp < hm)
            {
                yp++;
                srcPtr += widthElem;
            }

            ++sp;
            if (sp >= stackLenH) sp = 0;
        }
    }
}

void stackBlurRef(const Mat& img, Mat& dst, Size ksize)
{
    if(img.depth() == CV_8U)
        _stackblurRef<uchar>(img, dst, ksize);
    else if (img.depth() == CV_16S)
        _stackblurRef<short>(img, dst, ksize);
    else if (img.depth() == CV_16U)
        _stackblurRef<ushort>(img, dst, ksize);
    else if (img.depth() == CV_32F)
        _stackblurRef<float>(img, dst, ksize);
    else
        CV_Error(Error::StsNotImplemented,
                   ("Unsupported Mat type in stackBlurRef, "
                    "the supported formats are: CV_8U, CV_16U, CV_16S and CV_32F."));
}

std::vector<Size> kernelSizeVec = {
                Size(3, 3),
                Size(5, 5),
                Size(101, 101),
                Size(3, 9)
        };

typedef testing::TestWithParam<tuple<int, int, int> > StackBlur;

TEST_P (StackBlur, regression)
{
    Mat img_ = imread(findDataFile("shared/fruits.png"), 1);
    const int cn = get<0>(GetParam());
    const int kIndex = get<1>(GetParam());
    const int dtype = get<2>(GetParam());

    Size ksize = kernelSizeVec[kIndex];

    Mat img, dstRef, dst;
    convert(img_, img, dtype);

    vector<Mat> channels;
    split(img, channels);
    channels.push_back(channels[0]); // channels size is 4.

    Mat imgCn;
    if (cn == 1)
        imgCn = channels[0];
    else if (cn == 4)
        merge(channels, imgCn);
    else
        imgCn = img;

    stackBlurRef(imgCn, dstRef, ksize);
    stackBlur(imgCn, dst, ksize);
    EXPECT_LE(cvtest::norm(dstRef, dst, NORM_INF), 2.);
}

INSTANTIATE_TEST_CASE_P(Imgproc, StackBlur,
                        testing::Combine(
                                testing::Values(1, 3, 4),
                                testing::Values(0, 1, 2, 3),
                                testing::Values(CV_8U, CV_16S, CV_16U, CV_32F)
                        )
);

typedef testing::TestWithParam<tuple<int> > StackBlur_GaussianBlur;

// StackBlur should produce similar results as GaussianBlur output.
TEST_P(StackBlur_GaussianBlur, compare)
{
    Mat img_ = imread(findDataFile("shared/fruits.png"), 1);
    const int dtype = get<0>(GetParam());

    Size ksize(3, 3);
    Mat img, dstS, dstG;
    convert(img_, img, dtype);

    stackBlur(img, dstS, ksize);
    GaussianBlur(img,  dstG, ksize, 0);

    EXPECT_LE(cvtest::norm(dstS, dstG, NORM_INF), 13.);
}

INSTANTIATE_TEST_CASE_P(Imgproc, StackBlur_GaussianBlur, testing::Values(CV_8U, CV_16S, CV_16U, CV_32F));
}
}
