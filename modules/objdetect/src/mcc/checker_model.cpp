// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "precomp.hpp"

#include "checker_model.hpp"
#include "dictionary.hpp"

namespace cv
{
namespace mcc
{

///////////////////////////////////////////////////////////////////////////////
/// CChartModel
CChartModel::CChartModel(const COLORCHART chartType)
{
    switch (chartType)
    {
    case MCC24: //Standard
        size = cv::Size2i(4, 6);
        boxsize = cv::Size2f(11.25, 16.75);
        box.resize(4);
        box[0] = cv::Point2f(0.00, 0.00);
        box[1] = cv::Point2f(16.75, 0.00);
        box[2] = cv::Point2f(16.75, 11.25);
        box[3] = cv::Point2f(0.00, 11.25);

        cellchart.assign(CChartClassicModelCellchart, CChartClassicModelCellchart + 4 * 24);
        center.assign(CChartClassicModelCenter, CChartClassicModelCenter + 24);

        chart.resize(size.area(), std::vector<float>(9));
        for(int color = 0 ; color < (int)chart.size() ; color++)
            chart[color].assign(CChartClassicModelColors[color] , CChartClassicModelColors[color] + 9 );

        break;
    case SG140: //DigitalSG
        size = cv::Size2i(10, 14);
        boxsize = cv::Size2f(27.75, 38.75);
        box.resize(4);
        box[0] = cv::Point2f(0.00, 0.00);
        box[1] = cv::Point2f(38.75, 0.00);
        box[2] = cv::Point2f(38.75, 27.75);
        box[3] = cv::Point2f(0.00, 27.75);

        cellchart.assign(CChartDigitalSGCellchart, CChartDigitalSGCellchart + 4 * 140);
        center.assign(CChartDigitalSGCenter, CChartDigitalSGCenter + 140);

        chart.resize(size.area(), std::vector<float>(9));
        for(int color = 0 ; color < (int)chart.size() ; color++)
            chart[color].assign(CChartDigitalSGColors[color] , CChartDigitalSGColors[color] + 9 );

        break;
    case VINYL18: //Vinyl
        size = cv::Size2i(3, 6);
        boxsize = cv::Size2f(12.50, 18.25);
        box.resize(4);
        box[0] = cv::Point2f(0.00, 0.00);
        box[1] = cv::Point2f(18.25, 0.00);
        box[2] = cv::Point2f(18.25, 12.50);
        box[3] = cv::Point2f(0.00, 12.50);

        cellchart.assign(CChartVinylCellchart, CChartVinylCellchart + 4 * 18);
        center.assign(CChartVinylCenter, CChartVinylCenter + 18);

        chart.resize(size.area(), std::vector<float>(9));
        for(int color = 0 ; color < (int)chart.size() ; color++)
            chart[color].assign(CChartVinylColors[color] , CChartVinylColors[color] + 9 );

        break;
    }
}

CChartModel::~CChartModel()
{
}

bool CChartModel::
    evaluate(const SUBCCMModel &subModel, int &offset, int &iTheta, float &error)
{

    float tError;
    int tTheta, tOffset;
    error = INFINITY;
    bool beval = false;

    // para todas las orientaciones
    // min_{ theta,dt } | CC_e - CC |
    for (tTheta = 0; tTheta < 8; tTheta++)
    {
        if (match(subModel, tTheta, tError, tOffset) && tError < error)
        {
            error = tError;
            iTheta = tTheta;
            offset = tOffset;
            beval = true;
        }
    }

    return beval;
}

void CChartModel::
    copyToColorMat(OutputArray lab, int cs)
{
    size_t N, M, k;

    N = size.width;
    M = size.height;
    cv::Mat im_lab_org((int)N, (int)M, CV_32FC3);
    int type_color = 3 * cs;
    k = 0;

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            cv::Vec3f &lab_values = im_lab_org.at<cv::Vec3f>((int)i, (int)j);
            lab_values[0] = chart[k][type_color + 0];
            lab_values[1] = chart[k][type_color + 1];
            lab_values[2] = chart[k][type_color + 2];
            k++;
        }
    }

    lab.assign(im_lab_org);
}

void mcc::CChartModel::
    rotate90()
{

    size = cv::Size2i(size.height, size.width);

    //the matrix is roated clockwise 90 degree, so first row will become last column, second row second last column and so on
    //doing this inplace will make the code a bit hard to read, so creating a temporary array

    std::vector<cv::Point2f> _cellchart(cellchart.size());
    std::vector<cv::Point2f> _center(center.size());

    int k = 0;
    for (int i = 0; i < size.width; i++)
    {
        for (int j = 0; j < size.height; j++)
        {
            //k contains the new coordintes,
            int old_i = size.height - j - 1;
            int old_j = i;
            int old_k = (old_i)*size.width + old_j;

            _cellchart[4 * k + 0] = cellchart[4 * old_k + 3];
            _cellchart[4 * k + 1] = cellchart[4 * old_k + 0];
            _cellchart[4 * k + 2] = cellchart[4 * old_k + 1];
            _cellchart[4 * k + 3] = cellchart[4 * old_k + 2];

            // center
            _center[k] = center[old_k];
            k++;
        }
    }
    cellchart = _cellchart;
    center = _center;

    boxsize = cv::Size2f(boxsize.height, boxsize.width);
}

void mcc::CChartModel::
    flip()
{

    std::vector<cv::Point2f> _cellchart(cellchart.size());
    std::vector<cv::Point2f> _center(center.size());

    int k = 0;
    for (int i = 0; i < size.width; i++)
    {
        for (int j = 0; j < size.height; j++)
        {
            //k contains the new coordintes,
            int old_i = i;
            int old_j = size.height - j - 1;
            int old_k = (old_i)*size.height + old_j;

            _cellchart[4 * k + 0] = cellchart[4 * old_k + 1];
            _cellchart[4 * k + 1] = cellchart[4 * old_k + 0];
            _cellchart[4 * k + 2] = cellchart[4 * old_k + 3];
            _cellchart[4 * k + 3] = cellchart[4 * old_k + 2];

            // center
            _center[k] = center[old_k];
            k++;
        }
    }
    cellchart = _cellchart;
    center = _center;
}

float CChartModel::
    dist_color_lab(InputArray lab1, InputArray lab2)
{

    int N = lab1.rows();
    float dist = 0, dist_i;

    Mat _lab1 = lab1.getMat(), _lab2 = lab2.getMat();
    for (int i = 0; i < N; i++)
    {
        cv::Vec3f v1 = _lab1.at<cv::Vec3f>(i, 0);
        cv::Vec3f v2 = _lab2.at<cv::Vec3f>(i, 0);
        // v1[0] = 1;
        // v2[0] = 1; // L <- 0

        // euclidean
        cv::Vec3f v = v1 - v2;
        dist_i = v.dot(v);
        dist += sqrt(dist_i);

        // cosine
        //float costh = v1.dot(v2) / (norm(v1)*norm(v2));
        //dist += 1 - (1 + costh) / 2;
    }

    dist /= N;
    return dist;
}

bool CChartModel::
    match(const SUBCCMModel &subModel, int iTheta, float &error, int &ierror)
{

    size_t N, M, k;

    N = size.width;
    M = size.height;
    cv::Mat im_lab_org((int)N, (int)M, CV_32FC3);
    int type_color = 3;
    k = 0;

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            cv::Vec3f &lab = im_lab_org.at<cv::Vec3f>((int)i, (int)j);
            lab[0] = chart[k][type_color + 0];
            lab[1] = chart[k][type_color + 1];
            lab[2] = chart[k][type_color + 2];
            k++;
        }
    }

    rot90(im_lab_org, iTheta);
    N = im_lab_org.rows;
    M = im_lab_org.cols;

    size_t n, m;
    n = subModel.color_size.height;
    m = subModel.color_size.width;

    // boundary condition
    if (N < n || M < m)
        return false;

    // rgb to La*b*
    cv::Mat rgb_est = subModel.sub_chart;
    cv::Mat lab_est;

    // RGB color space
    //cv::cvtColor(rgb_est, lab_est, COLOR_BGR2RGB);

    // Lab color space
    //rgb_est *= 1/255;
    cv::cvtColor(rgb_est, lab_est, COLOR_BGR2Lab);

    size_t nN, mM;
    nN = N - n + 1;
    mM = M - m + 1;
    std::vector<float> lEcm(nN * mM);
    k = 0;
    for (size_t i = 0; i < nN; i++)
    {
        for (size_t j = 0; j < mM; j++)
        {
            cv::Mat lab_curr, lab_roi;
            lab_roi = im_lab_org(cv::Rect((int)j, (int)i, (int)m, (int)n));
            lab_roi.copyTo(lab_curr);
            lab_curr = lab_curr.t();
            lab_curr = lab_curr.reshape(3, (int)n * (int)m);

            // Mean squared error
            // ECM = 1 / N sum_i(Y - Yp) ^ 2
            lEcm[k] = dist_color_lab(lab_curr, lab_est) / (M * N);
            k++;
        }
    }

    // minimo
    error = lEcm[0];
    ierror = 0;
    for (int i = 1; i < (int)lEcm.size(); i++)
        if (error > lEcm[i])
        {
            error = lEcm[i];
            ierror = i;
        }

    return true;
}

void CChartModel::
    rot90(InputOutputArray mat, int itheta)
{

    //1=CW, 2=CCW, 3=180
    switch (itheta)
    {
    case 1: //transpose+flip(1)=CW
        transpose(mat, mat);
        cv::flip(mat, mat, 1);
        break;
    case 2: //flip(-1)=180
        cv::flip(mat, mat, -1);
        break;
    case 3: //transpose+flip(0)=CCW
        transpose(mat, mat);
        cv::flip(mat, mat, 0);
        break;
    //flipped images start here
    case 4: //flip(1)=no rotation, just flipped
        cv::flip(mat, mat, 1);
        break;
    case 5: //flip(1)+transpose + flip(1)=CW
        cv::flip(mat, mat, 1);
        transpose(mat, mat);
        cv::flip(mat, mat, 1);
        break;
    case 6: //flip(1)+flip(-1)=180
        cv::flip(mat, mat, 1);
        cv::flip(mat, mat, -1);
        break;
    case 7: //flip(1)+transpose+flip(0)=CCW
        cv::flip(mat, mat, 1);
        transpose(mat, mat);
        cv::flip(mat, mat, 0);
        break;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// // CChecker

Ptr<CChecker> CChecker::create()
{
    return makePtr<CCheckerImpl>();
}

void CCheckerImpl::setTarget(COLORCHART _target)
{
    target = _target;
}
void CCheckerImpl::setBox(std::vector<Point2f> _box)
{
    box = _box;
}
void CCheckerImpl::setChartsRGB(Mat _chartsRGB)
{
    chartsRGB = _chartsRGB;
}
void CCheckerImpl::setChartsYCbCr(Mat _chartsYCbCr)
{
    chartsYCbCr = _chartsYCbCr;
}
void CCheckerImpl::setCost(float _cost)
{
    cost = _cost;
}
void CCheckerImpl::setCenter(Point2f _center)
{
    center = _center;
}

COLORCHART CCheckerImpl::getTarget()
{
    return target;
}
std::vector<Point2f> CCheckerImpl::getBox()
{
    return box;
}
std::vector<Point2f> CCheckerImpl::getColorCharts()
{
    // color chart classic model
    CChartModel cccm(getTarget());
    Mat lab;
    size_t N;
    std::vector<Point2f> fbox = cccm.box;
    std::vector<Point2f> cellchart = cccm.cellchart;
    std::vector<Point2f> charts(cellchart.size());

    // tranformation
    Matx33f ccT = getPerspectiveTransform(fbox, getBox());

    std::vector<Point2f> bch(4), bcht(4);
    N = cellchart.size() / 4;
    for (size_t i = 0, k; i < N; i++)
    {
        k = 4 * i;
        for (size_t j = 0ull; j < 4ull; j++)
            bch[j] = cellchart[k + j];

        polyanticlockwise(bch);
        transform_points_forward(ccT, bch, bcht);

        Point2f c(0, 0);
        for (size_t j = 0; j < 4; j++)
            c += bcht[j];
        c /= 4;
        for (size_t j = 0ull; j < 4ull; j++)
            charts[k+j] = ((bcht[j] - c) * 0.50) + c;
    }
    return charts;
}
Mat CCheckerImpl::getChartsRGB()
{
    return chartsRGB;
}
Mat CCheckerImpl::getChartsYCbCr()
{
    return chartsYCbCr;
}
float CCheckerImpl::getCost()
{
    return cost;
}
Point2f CCheckerImpl::getCenter()
{
    return center;
}

void transform_points_forward(const Matx33f& T, const std::vector<Point2f> &X, std::vector<Point2f> &Xt)
{
    size_t N = X.size();
    if (Xt.size() != N)
        Xt.resize(N);
    std::fill(Xt.begin(), Xt.end(), Point2f(0.f, 0.f));
    if (N == 0)
        return;

    Matx31f p, xt;
    Point2f pt;
    for (size_t i = 0; i < N; i++)
    {
        p(0, 0) = X[i].x;
        p(1, 0) = X[i].y;
        p(2, 0) = 1;
        xt = T * p;
        pt.x = xt(0, 0) / xt(2, 0);
        pt.y = xt(1, 0) / xt(2, 0);
        Xt[i] = pt;
    }
}

} // namespace mcc
} // namespace cv
