/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena Kuznetsova, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "fast_hough_transform.hpp"

#include <iostream>
using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates line finding with the Fast Hough transform.\n"
            "Usage:\n"
            "./fasthoughtransform\n"
            "<image_name>, default is 'building.jpg'\n"
            "<fht_image_depth>, default is " << CV_32S << "\n"
            "<fht_angle_range>, default is " << 6 << " (@see cv::AngleRangeOption)\n"
            "<fht_operator>, default is " << 2 << " (@see cv::HoughOp)\n"
            "<fht_makeskew>, default is " << 1 << "(@see cv::HoughDeskewOption)" << endl;
}

bool parseArgs(int argc, const char **argv,
               Mat &img,
               int &houghDepth,
               int &houghAngleRange,
               int &houghOperator,
               int &houghSkew)
{
    if (argc > 6)
    {
        cout << "Too many arguments" << endl;
        return false;
    }

    const char *filename = argc >= 2 ? argv[1] : "building.jpg";
    img = imread(filename, 0);
    if (img.empty())
    {
        cout << "Failed to load image from '" << filename << "'" << endl;
        return false;
    }

    houghDepth      = argc >= 3 ? atoi(argv[2]) : CV_32S;
    houghAngleRange = argc >= 4 ? atoi(argv[3]) : 6;//ARO_315_135
    houghOperator   = argc >= 5 ? atoi(argv[4]) : 2;//FHT_ADD
    houghSkew       = argc >= 6 ? atoi(argv[5]) : 1;//HDO_DESKEW

    return true;
}

bool getEdges(const Mat &src, Mat &dst)
{
    Mat ucharSingleSrc;
    src.convertTo(ucharSingleSrc, CV_8UC1);

    Canny(ucharSingleSrc, dst, 50, 200, 3);
    return true;
}

bool fht(const Mat &src, Mat &dst,
         int dstDepth, int angleRange, int op, int skew)
{
    FastHoughTransform(src, dst, dstDepth, angleRange, op, skew);
    return true;
}

template<typename T>
bool rel(pair<T, Point> const &a, pair<T, Point> const &b)
{
    return a.first > b.first;
}

template<typename T>
bool getLocalExtr(vector<Vec4i> &lines,
                  const Mat &src,
                  const Mat &fht,
                  float minWeight,
                  int maxCount)
{
    const int MAX_LEN = 10000;

    vector<pair<T, Point> > weightedPoints;
    for (int y = 0; y < fht.rows; ++y)
    {
        if (weightedPoints.size() > MAX_LEN)
            break;

        T const *pLine = (T *)fht.ptr(max(y - 1, 0));
        T const *cLine = (T *)fht.ptr(y);
        T const *nLine = (T *)fht.ptr(min(y + 1, fht.rows - 1));

        for (int x = 0; x < fht.cols; ++x)
        {
            if (weightedPoints.size() > MAX_LEN)
                break;

            T const value = cLine[x];
            if (value >= minWeight)
            {
                bool isLocalMax = true;
                for (int xx = max(x - 1, 0);
                     xx <= min(x + 1, fht.cols - 1);
                     ++xx)
                    isLocalMax &= (value >= cLine[xx] &&
                                   value >= pLine[xx] &&
                                   value >= nLine[xx]);
                if (isLocalMax)
                    weightedPoints.push_back(make_pair(value, Point(x, y)));
            }
        }
    }

    if (weightedPoints.empty())
        return true;

    sort(weightedPoints.begin(), weightedPoints.end(), &rel<T>);
    weightedPoints.resize(min(static_cast<int>(weightedPoints.size()),
                              maxCount));

    for (size_t i = 0; i < weightedPoints.size(); ++i)
    {
        Vec4i houghLine(0, 0, 0, 0);
        HoughPoint2Line(houghLine, weightedPoints[i].second, src);
        lines.push_back(houghLine);
    }
    return true;
}

bool getLocalExtr(vector<Vec4i> &lines,
                  const Mat &src,
                  const Mat &fht,
                  float minWeight,
                  int maxCount)
{
    int const depth = CV_MAT_DEPTH(fht.type());
    switch (depth)
    {
    case 0:
        return getLocalExtr<uchar>(lines, src, fht, minWeight, maxCount);
    case 1:
        return getLocalExtr<schar>(lines, src, fht, minWeight, maxCount);
    case 2:
        return getLocalExtr<ushort>(lines, src, fht, minWeight, maxCount);
    case 3:
        return getLocalExtr<short>(lines, src, fht, minWeight, maxCount);
    case 4:
        return getLocalExtr<int>(lines, src, fht, minWeight, maxCount);
   case 5:
        return getLocalExtr<float>(lines, src, fht, minWeight, maxCount);
    case 6:
        return getLocalExtr<double>(lines, src, fht, minWeight, maxCount);
    default:
        return false;
    }
}

void rescale(Mat const &src, Mat &dst,
                   int const maxHeight=500,
                   int const maxWidth = 1000)
{
    double scale = min(min(static_cast<double>(maxWidth) / src.cols,
                           static_cast<double>(maxHeight) / src.rows), 1.0);
    resize(src, dst, Size(), scale, scale, INTER_LINEAR);
}

void showHumanReadableImg(string const &name, Mat const &img)
{
    Mat ucharImg;
    img.convertTo(ucharImg, CV_MAKETYPE(CV_8U, img.channels()));
    rescale(ucharImg, ucharImg);
    imshow(name, ucharImg);
}

void showFht(Mat const &fht)
{
    double minv(0), maxv(0);
    minMaxLoc(fht, &minv, &maxv);
    Mat ucharFht;
    fht.convertTo(ucharFht, CV_MAKETYPE(CV_8U, fht.channels()),
                  255.0 / (maxv + minv), minv / (maxv + minv));
    rescale(ucharFht, ucharFht);
    imshow("fast hough transform", ucharFht);
}

void showLines(Mat const &src, vector<Vec4i> const &lines)
{
    Mat bgrSrc;
    cvtColor(src, bgrSrc, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); ++i)
    {
        Vec4i const &l = lines[i];
        line(bgrSrc, Point(l[0], l[1]), Point(l[2], l[3]),
             Scalar(0, 0, 255), 1, CV_AA);
    }

    rescale(bgrSrc, bgrSrc);
    imshow("lines", bgrSrc);
}

int main(int argc, const char **argv)
{
    Mat src;
    int depth(0);
    int angleRange(0);
    int op(0);
    int skew(0);

    if (!parseArgs(argc, argv, src, depth, angleRange, op, skew))
    {
        help();
        return -1;
    }
    showHumanReadableImg("src", src);

    Mat canny;
    if (!getEdges(src, canny))
    {
        cout << "Failed to select canny edges";
        return -2;
    }
    showHumanReadableImg("canny", canny);

    Mat hough;
    if (!fht(canny, hough, depth, angleRange, op, skew))
    {
        cout << "Failed to compute Fast Hough Transform";
        return -2;
    }
    showFht(hough);

    vector<Vec4i> lines;
    if (!getLocalExtr(lines, canny, hough,
                      static_cast<float>(255 * 0.3 * min(src.rows, src.cols)),
                      50))
    {
        cout << "Failed to find local maximums on FHT image";
        return -2;
    }
    showLines(canny, lines);

    waitKey();

    return 0;
}
